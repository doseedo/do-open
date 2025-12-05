#!/usr/bin/env python3
"""
HYBRID ARRANGEMENT GENERATOR

Pipeline:
1. Transformer generates coherent melody sequences
2. Rule-based orchestration derives bass, chords, and accompaniment

This approach works better than pure transformer arrangement because:
- Transformer excels at: melodic coherence, pattern continuation
- Rules excel at: harmonic voice leading, role-appropriate registers

Usage:
    python generate_hybrid.py --num-samples 5 --bars 16 --output arrangements.jsonl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import random
from pathlib import Path


# ============================================================================
# MELODY MODEL (from training)
# ============================================================================

class MelodyGPT(nn.Module):
    """Simple GPT for melody generation."""

    def __init__(self, vocab_size, block_size=512, n_layer=6, n_head=6, n_embd=384, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=n_embd * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ============================================================================
# ORCHESTRATION RULES
# ============================================================================

ORCHESTRATION_RULES = {
    'melody_to_bass': [
        {'transpose_delta': 0, 'octave_delta': -2, 'weight': 0.4},   # Root
        {'transpose_delta': 7, 'octave_delta': -2, 'weight': 0.35},  # Fifth below
        {'transpose_delta': 5, 'octave_delta': -2, 'weight': 0.25},  # Fourth below
    ],
    'melody_to_chord': [
        {'transpose_deltas': [0, 4, 7], 'octave_delta': -1, 'weight': 0.5},    # Major triad
        {'transpose_deltas': [0, 3, 7], 'octave_delta': -1, 'weight': 0.3},    # Minor triad
        {'transpose_deltas': [0, 4, 7, 11], 'octave_delta': -1, 'weight': 0.2}, # Maj7
    ],
    'melody_to_strings': [
        {'transpose_delta': 0, 'octave_delta': 0, 'weight': 0.5},
        {'transpose_delta': 7, 'octave_delta': -1, 'weight': 0.3},
        {'transpose_delta': 4, 'octave_delta': 0, 'weight': 0.2},
    ],
}


def select_rule(rules):
    """Select rule based on weights."""
    total = sum(r['weight'] for r in rules)
    r = random.random() * total
    cumsum = 0
    for rule in rules:
        cumsum += rule['weight']
        if r <= cumsum:
            return rule
    return rules[-1]


# ============================================================================
# MELODY GENERATION
# ============================================================================

def generate_melody(model, vocab, idx_to_token, num_bars=16, temperature=0.9, top_k=50, top_p=0.95):
    """Generate a melody sequence using the transformer."""
    model.eval()
    device = next(model.parameters()).device

    # Start with BAR token
    bar_idx = vocab.get('BAR', 0)
    generated = [bar_idx]

    bar_count = 1  # We already have one BAR token
    max_tokens = num_bars * 50  # Allow plenty of room

    with torch.no_grad():
        while len(generated) < max_tokens and bar_count <= num_bars:
            # Prepare input
            context = generated[-model.block_size:]
            x = torch.tensor([context], dtype=torch.long, device=device)

            # Get logits
            logits = model(x)
            logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_idx)

            # Track bars - stop after we have enough
            token_str = idx_to_token.get(next_idx, '')
            if token_str == 'BAR':
                bar_count += 1
                if bar_count > num_bars:
                    break

    # Convert to tokens
    tokens = [idx_to_token.get(idx, f'UNK_{idx}') for idx in generated]
    return tokens


# ============================================================================
# MELODY PARSING
# ============================================================================

def is_transpose_token(t):
    """Check if token is a transpose token (T0-T11) not a track token (TR:...)."""
    if not t.startswith('T'):
        return False
    if t.startswith('TR:'):
        return False
    rest = t[1:]
    if rest.lstrip('-').isdigit():
        return True
    return False


def parse_melody(tokens):
    """Parse melody tokens into structured events."""
    events = []
    current_bar = 0
    current_delta = 0

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t == 'BAR':
            current_bar += 1
        elif t.startswith('D') and t[1:].lstrip('-').isdigit():
            current_delta = int(t[1:])
        elif t.startswith('TR:') or t.startswith('ROLE:'):
            # Skip track and role tokens - we'll assign roles in orchestration
            pass
        elif t.startswith('P') and t[1:].isdigit():
            pattern_id = t
            transpose = 0
            octave = 0

            # Look ahead for T (transpose, not track) and O
            if i + 1 < len(tokens) and is_transpose_token(tokens[i + 1]):
                try:
                    transpose = int(tokens[i + 1][1:])
                    i += 1
                except:
                    pass
            if i + 1 < len(tokens) and tokens[i + 1].startswith('O'):
                try:
                    octave = int(tokens[i + 1][1:])
                    i += 1
                except:
                    pass

            events.append({
                'bar': current_bar,
                'pattern': pattern_id,
                'transpose': transpose,
                'octave': octave,
                'delta': current_delta,
            })
        i += 1

    return events


# ============================================================================
# ACCOMPANIMENT DERIVATION
# ============================================================================

def derive_bass(melody_events, rules=None):
    """Derive bass line from melody - plays on downbeats."""
    if rules is None:
        rules = ORCHESTRATION_RULES['melody_to_bass']

    bass_events = []
    bars_seen = set()

    for evt in melody_events:
        bar = evt['bar']
        if bar in bars_seen:
            continue
        bars_seen.add(bar)

        rule = select_rule(rules)
        bass_events.append({
            'bar': bar,
            'pattern': evt['pattern'],
            'transpose': (evt['transpose'] + rule['transpose_delta']) % 12,
            'octave': evt['octave'] + rule['octave_delta'],
            'delta': 0,
            'role': 'bass',
        })

    return bass_events


def derive_chords(melody_events, rules=None):
    """Derive chord hits from melody - triads on downbeats."""
    if rules is None:
        rules = ORCHESTRATION_RULES['melody_to_chord']

    chord_events = []
    bars_seen = set()

    for evt in melody_events:
        bar = evt['bar']
        if bar in bars_seen:
            continue
        bars_seen.add(bar)

        rule = select_rule(rules)

        for i, t_delta in enumerate(rule['transpose_deltas']):
            chord_events.append({
                'bar': bar,
                'pattern': evt['pattern'],
                'transpose': (evt['transpose'] + t_delta) % 12,
                'octave': evt['octave'] + rule['octave_delta'],
                'delta': 0,
                'role': 'piano',
                'voice': i,
            })

    return chord_events


def derive_strings(melody_events, rules=None, interval=2):
    """Derive string pad - sustained notes changing every N bars."""
    if rules is None:
        rules = ORCHESTRATION_RULES['melody_to_strings']

    string_events = []
    last_bar = -interval

    for evt in melody_events:
        bar = evt['bar']
        if bar - last_bar < interval:
            continue
        last_bar = bar

        rule = select_rule(rules)
        string_events.append({
            'bar': bar,
            'pattern': evt['pattern'],
            'transpose': (evt['transpose'] + rule['transpose_delta']) % 12,
            'octave': evt['octave'] + rule['octave_delta'],
            'delta': 0,
            'role': 'strings',
        })

    return string_events


# ============================================================================
# ARRANGEMENT ASSEMBLY
# ============================================================================

def assemble_arrangement(melody_events, bass_events, chord_events, string_events=None):
    """Combine all parts into interleaved token sequence."""
    all_events = []

    # Add melody with role
    for evt in melody_events:
        all_events.append({**evt, 'role': 'reed', 'priority': 0})  # Melody on reed

    # Add accompaniment
    for evt in bass_events:
        all_events.append({**evt, 'priority': 1})
    for evt in chord_events:
        all_events.append({**evt, 'priority': 2})
    if string_events:
        for evt in string_events:
            all_events.append({**evt, 'priority': 3})

    # Sort by bar, then priority, then voice
    all_events.sort(key=lambda x: (x['bar'], x['priority'], x.get('voice', 0)))

    # Generate tokens
    tokens = []
    current_bar = -1
    current_role = None

    for evt in all_events:
        if evt['bar'] > current_bar:
            current_bar = evt['bar']
            tokens.append('BAR')

        role = evt['role']
        if role != current_role:
            tokens.append(f"ROLE:{role}")
            current_role = role

        tokens.append(f"D{evt['delta']}")
        tokens.append(evt['pattern'])
        tokens.append(f"T{evt['transpose']}")
        tokens.append(f"O{evt['octave']}")

    return tokens


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_melody_model(model_path, vocab_path, device='cuda'):
    """Load trained melody model."""
    # Load vocab
    with open(vocab_path) as f:
        vocab = json.load(f)
    idx_to_token = {v: k for k, v in vocab.items()}

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get model config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    vocab_size = len(vocab)

    model = MelodyGPT(
        vocab_size=vocab_size,
        block_size=config.get('block_size', 512),
        n_layer=config.get('n_layer', 6),
        n_head=config.get('n_head', 6),
        n_embd=config.get('n_embd', 384),
    ).to(device)

    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Loaded model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    return model, vocab, idx_to_token


def generate_arrangement(model, vocab, idx_to_token, num_bars=16,
                         include_bass=True, include_chords=True, include_strings=False,
                         temperature=0.9):
    """Generate a complete multi-track arrangement."""

    # 1. Generate melody with transformer
    melody_tokens = generate_melody(
        model, vocab, idx_to_token,
        num_bars=num_bars,
        temperature=temperature
    )

    # 2. Parse melody into events
    melody_events = parse_melody(melody_tokens)

    if not melody_events:
        return melody_tokens, {'melody_events': 0}

    # 3. Derive accompaniment using rules
    bass_events = derive_bass(melody_events) if include_bass else []
    chord_events = derive_chords(melody_events) if include_chords else []
    string_events = derive_strings(melody_events) if include_strings else []

    # 4. Assemble full arrangement
    arrangement_tokens = assemble_arrangement(
        melody_events, bass_events, chord_events, string_events
    )

    stats = {
        'melody_events': len(melody_events),
        'bass_events': len(bass_events),
        'chord_events': len(chord_events),
        'string_events': len(string_events),
        'total_tokens': len(arrangement_tokens),
        'bars': num_bars,
    }

    return arrangement_tokens, stats


def main():
    parser = argparse.ArgumentParser(description='Hybrid Arrangement Generator')
    parser.add_argument('--model', default='music_gpt_melody_v5.pt', help='Melody model checkpoint')
    parser.add_argument('--vocab', default='vocab_melody_v5.json', help='Vocabulary file')
    parser.add_argument('--output', '-o', default='hybrid_arrangements.jsonl', help='Output file')
    parser.add_argument('--num-samples', '-n', type=int, default=5, help='Number of arrangements')
    parser.add_argument('--bars', type=int, default=16, help='Bars per arrangement')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--no-bass', action='store_true', help='Disable bass')
    parser.add_argument('--no-chords', action='store_true', help='Disable chords')
    parser.add_argument('--with-strings', action='store_true', help='Include strings')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("HYBRID ARRANGEMENT GENERATOR")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Generating: {args.num_samples} arrangements, {args.bars} bars each")
    print(f"Parts: melody + {'bass ' if not args.no_bass else ''}{'chords ' if not args.no_chords else ''}{'strings' if args.with_strings else ''}")
    print("=" * 60)

    # Resolve paths
    base_dir = Path(__file__).parent
    model_path = base_dir / args.model
    vocab_path = base_dir / args.vocab

    # Load model
    model, vocab, idx_to_token = load_melody_model(model_path, vocab_path, args.device)

    # Generate arrangements
    results = []
    for i in range(args.num_samples):
        print(f"\nGenerating arrangement {i+1}/{args.num_samples}...")

        tokens, stats = generate_arrangement(
            model, vocab, idx_to_token,
            num_bars=args.bars,
            include_bass=not args.no_bass,
            include_chords=not args.no_chords,
            include_strings=args.with_strings,
            temperature=args.temperature,
        )

        results.append({
            'id': f'hybrid_{i}',
            'tokens': tokens,
            'stats': stats,
        })

        print(f"  Melody: {stats['melody_events']} events")
        print(f"  Bass: {stats['bass_events']} events")
        print(f"  Chords: {stats['chord_events']} events")
        print(f"  Total: {stats['total_tokens']} tokens")
        print(f"  Preview: {' '.join(tokens[:60])}...")

    # Save
    output_path = base_dir / args.output
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} arrangements to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
