#!/usr/bin/env python3
"""
Minimal Transformer for Music Token Sequences
==============================================

A small GPT-style model trained on pattern token sequences.
Learns: "What pattern/transform comes next?"

Architecture: 6 layers, 384 dim, 6 heads (~10M params)

Usage:
    python scripts/train_transformer.py corpus_tokens.jsonl vocab.json --epochs 10
    python scripts/train_transformer.py corpus_tokens.jsonl vocab.json --generate --checkpoint model.pt
"""

import json
import math
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Model
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MusicGPT(nn.Module):
    def __init__(self, vocab_size, block_size=512, n_layer=6, n_head=6, n_embd=384, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)

        # Count params
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================================
# Dataset
# ============================================================================

class TokenDataset(Dataset):
    def __init__(self, data_path, vocab, block_size=512):
        self.block_size = block_size
        self.vocab = vocab
        self.pad_id = vocab.get('<pad>', 0)
        self.bos_id = vocab.get('<bos>', 1)
        self.eos_id = vocab.get('<eos>', 2)

        # Load all sequences
        self.sequences = []
        with open(data_path) as f:
            for line in f:
                data = json.loads(line)
                tokens = data['tokens']
                # Convert to IDs
                ids = [self.bos_id] + [vocab.get(t, vocab.get('<unk>', 3)) for t in tokens] + [self.eos_id]
                self.sequences.append(ids)

        print(f"Loaded {len(self.sequences)} sequences")

        # Create training chunks
        self.chunks = []
        for seq in self.sequences:
            for i in range(0, len(seq) - 1, block_size // 2):  # 50% overlap
                chunk = seq[i:i + block_size + 1]
                if len(chunk) > 10:  # Skip very short chunks
                    self.chunks.append(chunk)

        print(f"Created {len(self.chunks)} training chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        # Pad if needed
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [self.pad_id] * (self.block_size + 1 - len(chunk))

        chunk = chunk[:self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================================
# Training
# ============================================================================

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    return total_loss / n_batches


def generate_tokens(model, vocab, device, prompt_tokens=None, length=200, temperature=0.9, top_k=50):
    """Generate token sequence."""
    model.eval()

    # Reverse vocab for decoding
    id_to_token = {v: k for k, v in vocab.items()}

    # Start with BOS or prompt
    if prompt_tokens:
        ids = [vocab.get('<bos>', 1)] + [vocab.get(t, vocab.get('<unk>', 3)) for t in prompt_tokens]
    else:
        ids = [vocab.get('<bos>', 1)]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        idx = model.generate(idx, max_new_tokens=length, temperature=temperature, top_k=top_k)

    # Decode
    generated_ids = idx[0].tolist()
    tokens = [id_to_token.get(i, '<unk>') for i in generated_ids]

    # Remove special tokens
    tokens = [t for t in tokens if t not in ['<pad>', '<bos>', '<eos>', '<unk>']]

    return tokens


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train transformer on music tokens')
    parser.add_argument('data', help='Path to corpus_tokens.jsonl')
    parser.add_argument('vocab', help='Path to vocab.json')
    parser.add_argument('--checkpoint', '-c', default='music_gpt.pt', help='Model checkpoint path')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--block-size', type=int, default=256, help='Context length')
    parser.add_argument('--generate', '-g', action='store_true', help='Generate instead of train')
    parser.add_argument('--length', '-l', type=int, default=200, help='Generation length')
    parser.add_argument('--temperature', '-t', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--output', '-o', default='generated_tokens.txt', help='Output file for generated tokens')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocab
    with open(args.vocab) as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Create model
    model = MusicGPT(
        vocab_size=len(vocab),
        block_size=args.block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    ).to(device)

    if args.generate:
        # Load checkpoint and generate
        if Path(args.checkpoint).exists():
            print(f"Loading checkpoint: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        else:
            print(f"WARNING: No checkpoint found at {args.checkpoint}, using random weights")

        print(f"\nGenerating {args.length} tokens...")
        tokens = generate_tokens(model, vocab, device, length=args.length, temperature=args.temperature)

        # Save
        with open(args.output, 'w') as f:
            f.write(' '.join(tokens))

        print(f"Saved to: {args.output}")
        print(f"\nFirst 50 tokens: {' '.join(tokens[:50])}")

    else:
        # Train
        print(f"\nLoading data: {args.data}")
        dataset = TokenDataset(args.data, vocab, block_size=args.block_size)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        print(f"\nTraining for {args.epochs} epochs...")
        best_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer, device, epoch)
            scheduler.step()

            print(f"Epoch {epoch} | Avg Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # Save checkpoint
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), args.checkpoint)
                print(f"  Saved best model to {args.checkpoint}")

            # Generate sample
            if epoch % 2 == 0:
                sample = generate_tokens(model, vocab, device, length=50, temperature=0.9)
                print(f"  Sample: {' '.join(sample[:30])}...")

        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        print(f"Model saved to: {args.checkpoint}")


if __name__ == '__main__':
    main()
