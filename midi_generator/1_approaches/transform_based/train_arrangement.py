#!/usr/bin/env python3
"""
ARRANGEMENT TRANSFORMER
========================

Train a transformer to generate accurate multi-track arrangements.

Key improvements over basic training:
1. Uses full multi-track corpus (all roles: melody, bass, chords, etc.)
2. Role-aware positional encoding (learns role transitions)
3. Larger model capacity for complex arrangements
4. Bar-aware attention bias (stronger connections within bars)
5. Gradient accumulation for effective larger batch sizes

Usage:
    python train_arrangement.py --epochs 50 --batch-size 16
    python train_arrangement.py --generate --checkpoint arrangement_gpt.pt
"""

import json
import math
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Model Architecture
# ============================================================================

class RoleAwareSelfAttention(nn.Module):
    """Self-attention with role embedding and bar-aware bias."""

    def __init__(self, n_embd, n_head, block_size, n_roles=16, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Role-based attention bias
        self.role_bias = nn.Parameter(torch.zeros(n_roles, n_roles))

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x, role_ids=None):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Add role bias if provided
        if role_ids is not None:
            role_bias = self.role_bias[role_ids.unsqueeze(2), role_ids.unsqueeze(1)]  # [B, T, T]
            att = att + role_bias.unsqueeze(1)  # [B, n_head, T, T]

        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, expansion=4):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, expansion * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(expansion * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, n_roles=16, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = RoleAwareSelfAttention(n_embd, n_head, block_size, n_roles, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, role_ids=None):
        x = x + self.attn(self.ln_1(x), role_ids)
        x = x + self.mlp(self.ln_2(x))
        return x


class ArrangementGPT(nn.Module):
    """
    GPT model specialized for multi-track music arrangements.

    Features:
    - Role-aware attention (learns interactions between bass, melody, chords, etc.)
    - Larger embedding dimension for capturing complex patterns
    - Deeper network for better arrangement understanding
    """

    def __init__(self, vocab_size, block_size=512, n_layer=8, n_head=8, n_embd=512,
                 n_roles=16, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_roles = n_roles

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            wre=nn.Embedding(n_roles + 1, n_embd),  # Role embedding (+1 for unknown)
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(n_embd, n_head, block_size, n_roles, dropout)
                             for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

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

    def forward(self, idx, role_ids=None, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        # Add role embedding if provided
        if role_ids is not None:
            role_emb = self.transformer.wre(role_ids)
            x = self.transformer.drop(tok_emb + pos_emb + role_emb)
        else:
            x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x, role_ids)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=0, label_smoothing=0.1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, role_ids=None, max_new_tokens=200, temperature=1.0,
                 top_k=None, top_p=0.95, rep_penalty=1.3, rep_window=50,
                 id_to_token=None):
        """Generate with nucleus sampling, role tracking, and repetition penalty.

        Args:
            idx: Starting token indices
            role_ids: Role IDs for each token
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (default 1.0)
            top_k: Top-k filtering (default None, use top_p instead)
            top_p: Nucleus sampling threshold (default 0.95)
            rep_penalty: Repetition penalty for recent patterns (default 1.3)
            rep_window: Window size for tracking recent tokens (default 50)
            id_to_token: Optional dict mapping token IDs to strings for penalty
        """

        # Role mapping for generation
        role_to_id = {
            'piano': 0, 'guitar': 1, 'bass': 2, 'strings': 3, 'brass': 4,
            'reed': 5, 'pipe': 6, 'synlead': 7, 'synpad': 8, 'synfx': 9,
            'organ': 10, 'chromperc': 11, 'ethnic': 12, 'percussive': 13,
            'soundfx': 14, 'melody': 15
        }

        current_role = 0  # Default role

        # Track recent tokens for repetition penalty
        recent_tokens = []

        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            role_cond = role_ids[:, -idx_cond.size(1):] if role_ids is not None else None

            # Forward pass
            logits, _ = self(idx_cond, role_cond)
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty to patterns, but NOT within orchestration windows
            # Orchestration = same pattern on multiple tracks within same bar (should NOT be penalized)
            # Repetition = same pattern reused across bars (SHOULD be penalized, but gently)
            if rep_penalty > 1.0 and recent_tokens and id_to_token is not None:
                from collections import Counter

                # Find the last BAR token position to identify cross-bar repetition
                last_bar_pos = None
                for i, tok_id in enumerate(reversed(recent_tokens[-rep_window:])):
                    if id_to_token.get(tok_id, '') == 'BAR':
                        last_bar_pos = len(recent_tokens) - i
                        break

                # Only count patterns BEFORE the last bar (cross-bar repetition)
                if last_bar_pos is not None:
                    tokens_before_bar = recent_tokens[:last_bar_pos]
                else:
                    tokens_before_bar = []

                token_counts = Counter(tokens_before_bar[-rep_window:])
                for token_id, count in token_counts.items():
                    tok_str = id_to_token.get(token_id, '')
                    if tok_str.startswith('P') and tok_str[1:].isdigit():
                        # Gentle penalty for cross-bar repetition only
                        penalty = rep_penalty ** min(count, 2)
                        logits[0, token_id] = logits[0, token_id] / penalty

            # Top-k filtering (optional, disabled by default)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Track token for repetition penalty
            recent_tokens.append(idx_next.item())

            # Update role tracking
            if role_ids is not None:
                role_ids = torch.cat((role_ids, torch.tensor([[current_role]], device=idx.device)), dim=1)

        return idx


# ============================================================================
# Dataset
# ============================================================================

ROLE_MAP = {
    'piano': 0, 'guitar': 1, 'bass': 2, 'strings': 3, 'brass': 4,
    'reed': 5, 'pipe': 6, 'synlead': 7, 'synpad': 8, 'synfx': 9,
    'organ': 10, 'chromperc': 11, 'ethnic': 12, 'percussive': 13,
    'soundfx': 14, 'melody': 15
}


class ArrangementDataset(Dataset):
    """Dataset for multi-track arrangements with role tracking."""

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

                # Convert to IDs and extract roles
                ids = [self.bos_id]
                roles = [0]  # Default role for BOS
                current_role = 0

                for token in tokens:
                    # Track role changes
                    if token.startswith('ROLE:'):
                        role_name = token.split(':')[1]
                        current_role = ROLE_MAP.get(role_name, 0)

                    token_id = vocab.get(token, vocab.get('<unk>', 3))
                    ids.append(token_id)
                    roles.append(current_role)

                ids.append(self.eos_id)
                roles.append(0)  # Default role for EOS

                self.sequences.append((ids, roles))

        print(f"Loaded {len(self.sequences)} sequences")

        # Create training chunks with overlap
        self.chunks = []
        for ids, roles in self.sequences:
            stride = block_size // 2  # 50% overlap
            for i in range(0, len(ids) - 1, stride):
                chunk_ids = ids[i:i + block_size + 1]
                chunk_roles = roles[i:i + block_size + 1]
                if len(chunk_ids) > 20:  # Skip very short chunks
                    self.chunks.append((chunk_ids, chunk_roles))

        print(f"Created {len(self.chunks)} training chunks")

        # Compute role distribution
        role_counts = defaultdict(int)
        for _, roles in self.chunks:
            for r in roles:
                role_counts[r] += 1

        print("Role distribution:")
        id_to_role = {v: k for k, v in ROLE_MAP.items()}
        for role_id, count in sorted(role_counts.items()):
            role_name = id_to_role.get(role_id, f'unknown_{role_id}')
            print(f"  {role_name}: {count}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids, roles = self.chunks[idx]

        # Pad if needed
        if len(ids) < self.block_size + 1:
            pad_len = self.block_size + 1 - len(ids)
            ids = ids + [self.pad_id] * pad_len
            roles = roles + [0] * pad_len

        ids = ids[:self.block_size + 1]
        roles = roles[:self.block_size + 1]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        r = torch.tensor(roles[:-1], dtype=torch.long)

        return x, y, r


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch,
                grad_accum_steps=4, max_grad_norm=1.0):
    """Train one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (x, y, r) in enumerate(train_loader):
        x, y, r = x.to(device), y.to(device), r.to(device)

        # Forward pass
        logits, loss = model(x, r, y)
        loss = loss / grad_accum_steps
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        if batch_idx % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item() * grad_accum_steps:.4f} | LR: {lr:.6f}")

    return total_loss / n_batches


def generate_sample(model, vocab, device, prompt_tokens=None, length=100,
                    temperature=0.7, top_k=None, top_p=0.92, rep_penalty=1.15):
    """Generate a sample arrangement with repetition penalty.

    Args:
        model: The trained model
        vocab: Token to ID mapping
        device: Device to run on
        prompt_tokens: Optional list of starting tokens
        length: Number of tokens to generate
        temperature: Sampling temperature (default 1.0)
        top_k: Top-k filtering (default None)
        top_p: Nucleus sampling threshold (default 0.95)
        rep_penalty: Repetition penalty (default 1.3)
    """
    model.eval()

    # Reverse vocab for decoding
    id_to_token = {v: k for k, v in vocab.items()}

    # Start with BOS
    if prompt_tokens:
        ids = [vocab.get('<bos>', 1)] + [vocab.get(t, vocab.get('<unk>', 3)) for t in prompt_tokens]
    else:
        ids = [vocab.get('<bos>', 1)]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    role_ids = torch.zeros_like(idx)

    # Generate (pass id_to_token for pattern-only repetition penalty)
    with torch.no_grad():
        idx = model.generate(idx, role_ids, max_new_tokens=length,
                            temperature=temperature, top_k=top_k, top_p=top_p,
                            rep_penalty=rep_penalty, id_to_token=id_to_token)

    # Decode
    generated_ids = idx[0].tolist()
    tokens = [id_to_token.get(i, '<unk>') for i in generated_ids]

    # Remove special tokens
    tokens = [t for t in tokens if t not in ['<pad>', '<bos>', '<eos>', '<unk>']]

    return tokens


def analyze_generation(tokens):
    """Analyze generated tokens for arrangement quality."""
    stats = {
        'n_bars': 0,
        'roles': defaultdict(int),
        'patterns': set(),
        'track_changes': 0,
    }

    current_role = None
    for t in tokens:
        if t == 'BAR':
            stats['n_bars'] += 1
        elif t.startswith('ROLE:'):
            role = t.split(':')[1]
            stats['roles'][role] += 1
            if role != current_role:
                stats['track_changes'] += 1
                current_role = role
        elif t.startswith('P'):
            stats['patterns'].add(t)

    return stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train arrangement transformer')
    parser.add_argument('--data', default='corpus_tokens_v4.jsonl', help='Corpus path')
    parser.add_argument('--vocab', default='vocab_v4.json', help='Vocab path')
    parser.add_argument('--checkpoint', '-c', default='arrangement_gpt.pt', help='Checkpoint path')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--block-size', type=int, default=512, help='Context length')
    parser.add_argument('--n-layer', type=int, default=8, help='Number of layers')
    parser.add_argument('--n-head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-embd', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--generate', '-g', action='store_true', help='Generate instead of train')
    parser.add_argument('--length', '-l', type=int, default=200, help='Generation length')
    parser.add_argument('--temperature', '-t', type=float, default=0.85, help='Sampling temperature')
    parser.add_argument('--output', '-o', default='generated_arrangement.jsonl', help='Output file')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Setup paths relative to script location
    script_dir = Path(__file__).parent
    data_path = script_dir / args.data
    vocab_path = script_dir / args.vocab
    checkpoint_path = script_dir / args.checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocab
    with open(vocab_path) as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Create model
    model = ArrangementGPT(
        vocab_size=len(vocab),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_roles=16,
        dropout=args.dropout,
    ).to(device)

    if args.generate:
        # Load checkpoint and generate
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Loaded epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('best_loss', '?'):.4f}")
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"WARNING: No checkpoint found at {checkpoint_path}")
            return

        print(f"\nGenerating {args.length} tokens...")

        # Generate multiple samples
        results = []
        for i in range(5):
            tokens = generate_sample(model, vocab, device, length=args.length,
                                    temperature=args.temperature)
            stats = analyze_generation(tokens)

            results.append({
                'sample_id': i,
                'tokens': tokens,
                'n_bars': stats['n_bars'],
                'n_patterns': len(stats['patterns']),
                'roles': dict(stats['roles']),
                'track_changes': stats['track_changes'],
            })

            print(f"\nSample {i}:")
            print(f"  Bars: {stats['n_bars']}, Patterns: {len(stats['patterns'])}")
            print(f"  Roles: {dict(stats['roles'])}")
            print(f"  Preview: {' '.join(tokens[:40])}...")

        # Save
        output_path = script_dir / args.output
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"\nSaved to: {output_path}")

    else:
        # Training mode
        print(f"\nLoading data: {data_path}")
        dataset = ArrangementDataset(data_path, vocab, block_size=args.block_size)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer with warmup
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * args.epochs // args.grad_accum
        warmup_steps = total_steps // 10  # 10% warmup

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Resume from checkpoint
        start_epoch = 1
        best_loss = float('inf')

        if args.resume and checkpoint_path.exists():
            print(f"Resuming from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 1) + 1
                best_loss = checkpoint.get('best_loss', float('inf'))
            else:
                model.load_state_dict(checkpoint)

        print(f"\nTraining for {args.epochs} epochs...")
        print(f"Effective batch size: {args.batch_size * args.grad_accum}")
        print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

        for epoch in range(start_epoch, args.epochs + 1):
            loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, epoch,
                grad_accum_steps=args.grad_accum
            )

            print(f"Epoch {epoch} | Avg Loss: {loss:.4f}")

            # Save checkpoint
            if loss < best_loss:
                best_loss = loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'vocab_size': len(vocab),
                    'config': {
                        'block_size': args.block_size,
                        'n_layer': args.n_layer,
                        'n_head': args.n_head,
                        'n_embd': args.n_embd,
                    }
                }, checkpoint_path)
                print(f"  Saved best model (loss: {best_loss:.4f})")

            # Generate sample every 5 epochs
            if epoch % 5 == 0:
                sample = generate_sample(model, vocab, device, length=80, temperature=0.85)
                stats = analyze_generation(sample)
                print(f"  Sample ({stats['n_bars']} bars, {len(stats['patterns'])} patterns):")
                print(f"    {' '.join(sample[:50])}...")

        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        print(f"Model saved to: {checkpoint_path}")


if __name__ == '__main__':
    main()
