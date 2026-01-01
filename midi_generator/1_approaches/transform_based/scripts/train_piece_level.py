#!/usr/bin/env python3
"""
Training Script for Piece-Level Pattern Model
==============================================

Trains a transformer to learn: what patterns follow what patterns in real music.

Philosophy: "Discovery not Prescription"
- Patterns are ALREADY discovered via Re-Pair compression
- The model learns their USAGE patterns in actual music pieces

Key considerations:
- Large vocabulary (~144K tokens) requires careful memory management
- Uses tied embeddings (input/output share weights) to reduce params
- Gradient accumulation for effective larger batch sizes
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from piece_level_dataset import create_dataloader, PieceLevelDataset


class PieceLevelTransformer(nn.Module):
    """
    Transformer for piece-level pattern generation.

    Uses tied embeddings to handle large vocabulary efficiently.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
        pad_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # Embedding (will be tied with output projection)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection (tied with embedding)
        # We'll use embedding weight transposed for output
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # Initialize
        self._init_weights()

        # Parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"  Embedding: {vocab_size * d_model:,}")
        print(f"  Transformer: {n_params - vocab_size * d_model:,}")

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """Forward pass with tied embeddings."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Padding mask
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Decoder
        x = self.decoder(
            tgt=x,
            memory=torch.zeros(batch_size, 1, self.d_model, device=device),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Final norm
        x = self.ln_f(x)

        # Output projection (tied weights)
        # logits = x @ W^T + b  where W = embedding weight
        logits = F.linear(x, self.token_embedding.weight, self.output_bias)

        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        device = prompt.device
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Truncate if too long
            if generated.size(1) > 1024:
                context = generated[:, -1024:]
            else:
                context = generated

            logits, _ = self.forward(context)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_id:
                break

        return generated


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch,
                log_interval=50, accumulation_steps=4):
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

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Count tokens
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * accumulation_steps * n_tokens
        total_tokens += n_tokens

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_tokens
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
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
def generate_sample(model, dataset, device, max_new_tokens=30):
    """Generate a sample sequence."""
    model.eval()

    # Start with BOS
    prompt = torch.tensor([[dataset.bos_id]], device=device)

    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        eos_id=dataset.eos_id,
    )

    # Decode
    id_to_token = {v: k for k, v in dataset.vocab.items()}
    tokens = [id_to_token.get(t.item(), '?') for t in generated[0]]
    return tokens


def main():
    parser = argparse.ArgumentParser(description='Train piece-level pattern model')
    parser.add_argument('--data-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/piece_level_tokens',
                        help='Path to tokenized data')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/checkpoints/piece_level',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--log-interval', type=int, default=20, help='Log every N batches')
    parser.add_argument('--save-interval', type=int, default=5, help='Save every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, dataset = create_dataloader(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=4,
        train_split=0.9,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = PieceLevelTransformer(
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
        'accumulation_steps': args.accumulation_steps,
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
        sample = generate_sample(model, dataset, device, max_new_tokens=20)
        # Show abbreviated form
        sample_str = ' '.join(s.replace('PATTERN_', 'P_').replace('TRACK_', 'T_').replace('BEAT_', 'B_')
                              for s in sample[:15])
        print(f"  Sample: {sample_str}...")

        # Save best
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
