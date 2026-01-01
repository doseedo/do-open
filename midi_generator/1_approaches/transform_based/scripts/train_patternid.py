"""
Train transformer on piece-level PATTERN ID sequences.
Each pattern is a single token - model learns which patterns go together.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
import json
from tqdm import tqdm
import os

class PatternIDDataset(Dataset):
    """Dataset for pattern ID sequences."""

    def __init__(self, sequences, max_length=512):
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Truncate if needed
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        # Pad
        padded = seq + [0] * (self.max_length - len(seq))
        return torch.tensor(padded, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatternIDTransformer(nn.Module):
    """Transformer for pattern ID sequence modeling."""

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, dropout=0.1, max_len=1024):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        seq_len = x.size(1)

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Padding mask
        pad_mask = (x == 0)

        # Embed and encode
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Transform
        x = self.transformer(x, mask=mask, src_key_padding_mask=pad_mask)

        # Project to vocab
        return self.fc_out(x)


def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_tokens = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        batch = batch.to(device)

        # Input and target (shifted by 1)
        x = batch[:, :-1]
        y = batch[:, 1:]

        # Forward
        logits = model(x)

        # Loss (ignore padding)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=0
        )

        # Scale loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Count non-padding tokens
        n_tokens = (y != 0).sum().item()
        total_loss += loss.item() * accumulation_steps * n_tokens
        total_tokens += n_tokens

        # Update weights
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({'loss': total_loss / max(total_tokens, 1)})

    return total_loss / max(total_tokens, 1)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=0,
                reduction='sum'
            )

            n_tokens = (y != 0).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='piece_level_patternid_data.pt')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/patternid')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    parser.add_argument('--val-split', type=float, default=0.1)
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    base = Path('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')
    data = torch.load(base / args.data_path)
    sequences = data['sequences']
    vocab_size = data['vocab_size']

    print(f"Loaded {len(sequences)} sequences")
    print(f"Vocab size: {vocab_size}")

    # Filter sequences that fit in context
    sequences = [s for s in sequences if len(s) <= args.max_length]
    print(f"Sequences fitting in {args.max_length} tokens: {len(sequences)}")

    # Split train/val
    n_val = int(len(sequences) * args.val_split)
    val_sequences = sequences[:n_val]
    train_sequences = sequences[n_val:]

    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}")

    # Datasets
    train_dataset = PatternIDDataset(train_sequences, args.max_length)
    val_dataset = PatternIDDataset(val_sequences, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = PatternIDTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_length
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.accumulation_steps

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return max(0.1, 1.0 - (step - args.warmup_steps) / (total_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoint dir
    ckpt_dir = base / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.accumulation_steps)
        val_loss = validate(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val perplexity: {math.exp(val_loss):.2f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_size': vocab_size,
            'args': vars(args)
        }

        torch.save(checkpoint, ckpt_dir / 'latest.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, ckpt_dir / 'best_model.pt')
            print(f"New best model saved!")

        # Early stopping check
        if math.exp(val_loss) < 2.0:
            print("Perplexity < 2.0, may be overfitting. Consider stopping.")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
