#!/usr/bin/env python3
"""
Train AutoVC-style Register Disentanglement Model

KEY TRAINING STRATEGY:
- Content from segment A
- Register from segment B (DIFFERENT segment, SAME pitch bin)
- Target: reconstruct segment A

This prevents the model from cheating by encoding formants in the content path.

Usage:
    python train_autovc.py --segments /path/to/segments.json --output_dir /path/to/output
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

from models_autovc import AutoVCLatent


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


class CrossSegmentDataset(Dataset):
    """
    Dataset that provides content/register pairs from DIFFERENT segments.

    Groups segments by pitch bin, then samples:
    - Content segment
    - Register segment (different segment, same pitch bin)
    """

    def __init__(self, segments_json: str, min_length: int = 64, max_length: int = 256,
                 pitch_bin_size: float = 2.0):
        self.min_length = min_length
        self.max_length = max_length
        self.pitch_bin_size = pitch_bin_size

        print(f"Loading segments from: {segments_json}")
        with open(segments_json) as f:
            data = json.load(f)

        # Collect segments grouped by pitch bin
        self.pitch_bins = defaultdict(list)

        for group_id, segs in data.get('segments_by_group', {}).items():
            for seg in segs:
                seg_len = seg['end_frame'] - seg['start_frame']
                if seg_len >= min_length:
                    midi = seg['median_midi']
                    pitch_bin = int(midi / pitch_bin_size)

                    self.pitch_bins[pitch_bin].append({
                        'latent_path': fix_path(seg['latent_path']),
                        'start_frame': seg['start_frame'],
                        'end_frame': seg['end_frame'],
                        'median_midi': midi,
                    })

        # Filter pitch bins with at least 2 segments (need pairs)
        valid_bins = {k: v for k, v in self.pitch_bins.items() if len(v) >= 2}
        self.pitch_bins = valid_bins

        # Create flat list of (pitch_bin, segment_idx) for sampling
        self.samples = []
        for pitch_bin, segments in self.pitch_bins.items():
            for idx in range(len(segments)):
                self.samples.append((pitch_bin, idx))

        print(f"Loaded {len(self.samples)} segments in {len(self.pitch_bins)} pitch bins")
        print(f"Pitch bin size: {pitch_bin_size} semitones")

        # Cache for loaded latents
        self.latent_cache = {}

    def __len__(self):
        return len(self.samples)

    def _load_latent(self, path):
        if path in self.latent_cache:
            return self.latent_cache[path]

        if not os.path.exists(path):
            return None

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

    def _extract_segment(self, seg_info, latent):
        """Extract and crop segment from latent."""
        start = seg_info['start_frame']
        end = min(seg_info['end_frame'], latent.shape[-1])
        seg_len = end - start

        # Clamp to max_length
        if seg_len > self.max_length:
            offset = random.randint(0, seg_len - self.max_length)
            start = start + offset
            end = start + self.max_length

        return latent[:, :, start:end].clone()

    def __getitem__(self, idx):
        pitch_bin, seg_idx = self.samples[idx]
        segments_in_bin = self.pitch_bins[pitch_bin]

        # Content segment
        content_seg = segments_in_bin[seg_idx]
        content_latent = self._load_latent(content_seg['latent_path'])

        if content_latent is None:
            # Fallback to random sample
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        content = self._extract_segment(content_seg, content_latent)

        # Register segment - DIFFERENT segment, same pitch bin
        other_indices = [i for i in range(len(segments_in_bin)) if i != seg_idx]
        register_idx = random.choice(other_indices)
        register_seg = segments_in_bin[register_idx]
        register_latent = self._load_latent(register_seg['latent_path'])

        if register_latent is None:
            register_latent = content_latent
            register_seg = content_seg

        register = self._extract_segment(register_seg, register_latent)

        return {
            'content': content,
            'register': register,
            'content_midi': torch.tensor(content_seg['median_midi']),
            'register_midi': torch.tensor(register_seg['median_midi']),
            'content_length': content.shape[-1],
            'register_length': register.shape[-1],
        }


def collate_fn(batch):
    """Collate with separate padding for content and register."""
    max_content_len = max(item['content'].shape[-1] for item in batch)
    max_register_len = max(item['register'].shape[-1] for item in batch)

    contents = []
    registers = []
    content_lengths = []
    register_lengths = []

    for item in batch:
        # Pad content
        c = item['content']
        if c.shape[-1] < max_content_len:
            c = F.pad(c, (0, max_content_len - c.shape[-1]))
        contents.append(c)
        content_lengths.append(item['content_length'])

        # Pad register
        r = item['register']
        if r.shape[-1] < max_register_len:
            r = F.pad(r, (0, max_register_len - r.shape[-1]))
        registers.append(r)
        register_lengths.append(item['register_length'])

    return {
        'content': torch.stack(contents),
        'register': torch.stack(registers),
        'content_lengths': torch.tensor(content_lengths),
        'register_lengths': torch.tensor(register_lengths),
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        content = batch['content'].to(device)
        register = batch['register'].to(device)
        content_lengths = batch['content_lengths']

        optimizer.zero_grad()

        # Forward: content from A, register from B, reconstruct A
        reconstructed, _, _ = model(content, register)

        # Reconstruction loss (masked for valid frames)
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
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        content = batch['content'].to(device)
        register = batch['register'].to(device)
        content_lengths = batch['content_lengths']

        reconstructed, _, _ = model(content, register)

        B, C, H, T = content.shape
        mask = torch.zeros(B, 1, 1, T, device=device)
        for i, length in enumerate(content_lengths):
            mask[i, :, :, :length] = 1.0

        loss = F.mse_loss(reconstructed * mask, content * mask)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def compute_register_embeddings_by_pitch(model, dataset, device, num_samples_per_bin=50):
    """
    Compute average register embeddings per pitch bin.
    """
    model.eval()

    embeddings_by_bin = defaultdict(list)

    for pitch_bin, segments in dataset.pitch_bins.items():
        samples = random.sample(segments, min(num_samples_per_bin, len(segments)))

        for seg in samples:
            latent = dataset._load_latent(seg['latent_path'])
            if latent is None:
                continue

            segment = dataset._extract_segment(seg, latent)
            segment = segment.unsqueeze(0).to(device)

            register_emb = model.encode_register(segment)
            embeddings_by_bin[pitch_bin].append(register_emb.cpu())

    # Average per bin
    avg_embeddings = {}
    for pitch_bin, embs in embeddings_by_bin.items():
        if embs:
            avg_embeddings[pitch_bin] = torch.cat(embs).mean(dim=0, keepdim=True)
            midi_center = pitch_bin * dataset.pitch_bin_size + dataset.pitch_bin_size / 2
            print(f"  Bin {pitch_bin} (MIDI ~{midi_center:.0f}): {len(embs)} samples")

    return avg_embeddings


def main():
    parser = argparse.ArgumentParser(description="Train AutoVC-style Register Disentanglement")
    parser.add_argument('--segments', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bottleneck_dim', type=int, default=32)
    parser.add_argument('--register_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--downsample', type=int, default=8)
    parser.add_argument('--pitch_bin_size', type=float, default=2.0,
                        help='Size of pitch bins in semitones')
    parser.add_argument('--min_length', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
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
    log("AutoVC-style Training with Cross-Segment Sampling")
    log("=" * 60)
    log(f"Bottleneck: {args.bottleneck_dim} × T/{args.downsample}")
    log(f"Register dim: {args.register_dim}")
    log(f"Pitch bin size: {args.pitch_bin_size} semitones")
    log(f"Training: Content from A, Register from B (same pitch), reconstruct A")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    # Dataset
    dataset = CrossSegmentDataset(
        args.segments,
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
    model = AutoVCLatent(
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        register_dim=args.register_dim,
        downsample=args.downsample,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {total_params:,}")

    # Optimizer
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

        # Save
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
    register_embeddings = compute_register_embeddings_by_pitch(model, dataset, device)

    torch.save({
        'embeddings': register_embeddings,
        'pitch_bin_size': args.pitch_bin_size,
    }, os.path.join(args.output_dir, 'register_embeddings.pt'))

    log(f"Saved {len(register_embeddings)} register embeddings")


if __name__ == "__main__":
    main()
