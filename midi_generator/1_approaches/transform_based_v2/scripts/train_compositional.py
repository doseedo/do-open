#!/usr/bin/env python3
"""
Training Script for Compositional Pattern Transformer
======================================================

Trains a transformer on piece-level compositional sequences.
Data format: BEAT/TRACK/OFFSET + INTERVAL+COMPOSE patterns

Can use either:
1. New piece-level data: piece_compositional_data.pt (has vocab embedded)
2. Old rule-level data: compositional_tokens_v2/ directory
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from compositional_model import create_model


class PieceCompositionalDataset(Dataset):
    """Dataset for piece-level compositional sequences (from .pt file)."""

    def __init__(self, sequences, vocab, max_length=1024, random_crop=True):
        self.sequences = sequences
        self.vocab = vocab
        self.max_length = max_length
        self.random_crop = random_crop
        self.vocab_size = len(vocab)
        self.pad_id = vocab.get('PAD', 0)
        self.bos_id = vocab.get('BOS', 1)
        self.eos_id = vocab.get('EOS', 2)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()

        # Random crop if too long
        if len(seq) > self.max_length:
            if self.random_crop:
                start = random.randint(0, len(seq) - self.max_length)
            else:
                start = 0
            seq = seq[start:start + self.max_length]

        input_ids = torch.tensor(seq, dtype=torch.long)
        return {
            'input_ids': input_ids[:-1],
            'labels': input_ids[1:],
            'length': len(input_ids) - 1,
        }


def collate_fn(batch, pad_id=0):
    """Collate with padding."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != pad_id).long()

    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
        'attention_mask': attention_mask,
    }


def create_dataloader_from_pt(data_path, batch_size=8, max_length=1024,
                              train_split=0.9, num_workers=4):
    """Create dataloaders from piece_compositional_data.pt file."""
    data = torch.load(data_path)
    sequences = data['sequences']
    vocab = data['vocab']
    id_to_token = data.get('id_to_token', {v: k for k, v in vocab.items()})
    vocab_size = data['vocab_size']

    # Split train/val
    n_train = int(len(sequences) * train_split)
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]

    train_dataset = PieceCompositionalDataset(train_seqs, vocab, max_length, random_crop=True)
    val_dataset = PieceCompositionalDataset(val_seqs, vocab, max_length, random_crop=False)

    pad_id = vocab.get('PAD', 0)
    collate = lambda batch: collate_fn(batch, pad_id)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, vocab, id_to_token


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch,
                log_interval=50, accumulation_steps=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(dtype=torch.bfloat16):
            logits, loss = model(input_ids, attention_mask, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Count non-padding tokens
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * accumulation_steps * n_tokens
        total_tokens += n_tokens

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_tokens
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            lr = scheduler.get_last_lr()[0]
            ppl = min(1e6, torch.exp(torch.tensor(avg_loss)).item())
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.0f}")

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(dtype=torch.bfloat16):
            logits, loss = model(input_ids, attention_mask, labels)

        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


@torch.no_grad()
def generate_sample(model, vocab, id_to_token, device, max_new_tokens=50):
    """Generate a sample sequence."""
    model.eval()

    # Start with BOS
    bos_id = vocab.get('BOS', 1)
    eos_id = vocab.get('EOS', 2)
    prompt = torch.tensor([[bos_id]], device=device)

    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        eos_id=eos_id,
    )

    # Decode tokens
    tokens = [id_to_token.get(t.item(), '?') for t in generated[0]]
    return tokens


def main():
    parser = argparse.ArgumentParser(description='Train compositional pattern model')
    parser.add_argument('--data-path', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/piece_compositional_data.pt',
                        help='Path to piece_compositional_data.pt file')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/checkpoints/compositional_piece',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--max-length', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--log-interval', type=int, default=20, help='Log every N batches')
    parser.add_argument('--save-interval', type=int, default=5, help='Save every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders from .pt file
    print("\nLoading data...")
    train_loader, val_loader, dataset, vocab, id_to_token = create_dataloader_from_pt(
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_split=0.9,
        num_workers=4,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Vocab size: {dataset.vocab_size}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=0.1,
        max_len=args.max_length + 10,
        pad_id=dataset.pad_id,
    ).to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    scaler = GradScaler()

    # Save config
    config = {
        'vocab_size': dataset.vocab_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_model * 4,
        'max_length': args.max_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args.log_interval, args.accumulation_steps
        )

        # Validate
        val_loss = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Generate sample
        sample = generate_sample(model, vocab, id_to_token, device, max_new_tokens=30)
        print(f"  Sample: {' '.join(sample[:20])}...")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, output_dir / 'best_model.pt')
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        # Periodic save
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, output_dir / f'checkpoint_epoch{epoch}.pt')

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
