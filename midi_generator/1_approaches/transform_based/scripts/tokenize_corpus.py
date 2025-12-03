#!/usr/bin/env python3
"""
Tokenize Corpus for Transformer Training
=========================================

Converts the pattern checkpoint into token sequences suitable for
training a causal language model (transformer).

The codec provides SEMANTIC TOKENS:
  - Pattern tokens: P123 (which musical pattern)
  - Transform tokens: T7 (what transposition)
  - Timing tokens: D480 (delta time in ticks)

The transformer learns SEQUENCE COHERENCE:
  - What pattern typically follows what?
  - What transforms are used together?
  - How do timing relationships work?

Output: JSONL file with one token sequence per track
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and return patterns + occurrences."""
    ckpt = np.load(checkpoint_path, allow_pickle=True)

    # Load patterns
    patterns_file = ckpt.get('patterns_json_file', [None])[0]
    if patterns_file:
        patterns_path = Path(checkpoint_path).parent / patterns_file
        with open(patterns_path) as f:
            rules = json.load(f)
    else:
        rules = json.loads(str(ckpt['patterns_json'][0]))

    return rules


def build_sequences(rules: dict):
    """Build token sequences from pattern occurrences."""
    # Group occurrences by (piece_id, track_id)
    sequences = defaultdict(list)

    for rule_id, rule in rules.items():
        pattern_len = len(rule.get('pitch_classes', []))

        for occ in rule.get('occurrences', []):
            piece_id = occ.get('piece_id', '')
            track_id = occ.get('track_id', 0)
            onset = occ.get('onset_time', 0)

            key = (piece_id, track_id)
            sequences[key].append({
                'rule_id': rule_id,
                'onset_time': onset,
                'pitch_offset': occ.get('pitch_offset', 0),
                'tau_offset': occ.get('tau_offset', 480),
                'octave': occ.get('octave_transform', 0) // 12,
                'pattern_len': pattern_len,
                'gm_program': occ.get('gm_program', 0),
            })

    # Sort each sequence by onset time
    for key in sequences:
        sequences[key].sort(key=lambda x: x['onset_time'])

    # Deduplicate: keep only longest pattern at each onset
    deduped_sequences = {}
    for key, occs in sequences.items():
        by_onset = defaultdict(list)
        for occ in occs:
            by_onset[occ['onset_time']].append(occ)

        filtered = []
        for onset_time in sorted(by_onset.keys()):
            best = max(by_onset[onset_time], key=lambda x: x['pattern_len'])
            filtered.append(best)

        if len(filtered) >= 8:  # Minimum sequence length
            deduped_sequences[key] = filtered

    return deduped_sequences


def tokenize_sequence(occs: list, include_timing: bool = True):
    """Convert occurrence sequence to tokens.

    Token format:
      P{rule_id}  - Pattern token (e.g., P123)
      T{offset}   - Transpose token (0-11)
      O{octave}   - Octave offset (-2 to +2)
      D{bucket}   - Delta time bucket (0-15)
    """
    tokens = []
    prev_onset = 0

    for i, occ in enumerate(occs):
        # Delta time (quantized to buckets)
        if include_timing and i > 0:
            delta = occ['onset_time'] - prev_onset
            # Bucket: 0=0, 1=120, 2=240, 3=480, 4=960, etc.
            if delta <= 0:
                bucket = 0
            elif delta <= 120:
                bucket = 1
            elif delta <= 240:
                bucket = 2
            elif delta <= 480:
                bucket = 3
            elif delta <= 960:
                bucket = 4
            elif delta <= 1920:
                bucket = 5
            elif delta <= 3840:
                bucket = 6
            else:
                bucket = 7
            tokens.append(f"D{bucket}")

        # Pattern token
        tokens.append(f"P{occ['rule_id']}")

        # Transform token (pitch offset mod 12)
        transpose = occ['pitch_offset'] % 12
        tokens.append(f"T{transpose}")

        # Octave token
        octave = max(-2, min(2, occ['octave']))
        tokens.append(f"O{octave}")

        prev_onset = occ['onset_time']

    return tokens


def build_vocab(all_tokens: list):
    """Build vocabulary from all tokens."""
    vocab = set()
    for tokens in all_tokens:
        vocab.update(tokens)

    # Sort for consistency
    vocab = sorted(vocab, key=lambda x: (x[0], int(x[1:]) if x[1:].lstrip('-').isdigit() else 0))

    # Add special tokens
    special = ['<pad>', '<bos>', '<eos>', '<unk>']
    vocab = special + vocab

    return {tok: i for i, tok in enumerate(vocab)}


def main():
    parser = argparse.ArgumentParser(description='Tokenize corpus for transformer training')
    parser.add_argument('checkpoint', help='Path to checkpoint .npz file')
    parser.add_argument('--output', '-o', default='corpus_tokens.jsonl', help='Output JSONL file')
    parser.add_argument('--vocab', '-v', default='vocab.json', help='Output vocab file')
    parser.add_argument('--no-timing', action='store_true', help='Exclude timing tokens')
    parser.add_argument('--min-length', type=int, default=16, help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    args = parser.parse_args()

    print("=" * 60)
    print("TOKENIZE CORPUS FOR TRANSFORMER")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    rules = load_checkpoint(args.checkpoint)
    print(f"  Loaded {len(rules)} patterns")

    # Build sequences
    print("\nBuilding sequences...")
    sequences = build_sequences(rules)
    print(f"  Found {len(sequences)} track sequences")

    # Tokenize
    print("\nTokenizing...")
    all_token_seqs = []
    include_timing = not args.no_timing

    for (piece_id, track_id), occs in sequences.items():
        tokens = tokenize_sequence(occs, include_timing=include_timing)

        if len(tokens) >= args.min_length:
            # Truncate if needed
            if len(tokens) > args.max_length:
                tokens = tokens[:args.max_length]

            all_token_seqs.append({
                'piece_id': piece_id,
                'track_id': track_id,
                'tokens': tokens,
                'n_patterns': len(occs),
            })

    print(f"  Tokenized {len(all_token_seqs)} sequences")

    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_vocab([s['tokens'] for s in all_token_seqs])
    print(f"  Vocabulary size: {len(vocab)}")

    # Stats
    total_tokens = sum(len(s['tokens']) for s in all_token_seqs)
    avg_len = total_tokens / len(all_token_seqs) if all_token_seqs else 0
    print(f"\nStats:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg sequence length: {avg_len:.1f}")
    print(f"  Unique patterns: {sum(1 for k in vocab if k.startswith('P'))}")

    # Save tokens
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for seq in all_token_seqs:
            f.write(json.dumps(seq) + '\n')
    print(f"\nSaved tokens to: {output_path}")

    # Save vocab
    vocab_path = Path(args.vocab)
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab to: {vocab_path}")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE SEQUENCE")
    print("=" * 60)
    if all_token_seqs:
        sample = all_token_seqs[0]
        print(f"Piece: {sample['piece_id']}, Track: {sample['track_id']}")
        print(f"Tokens ({len(sample['tokens'])}): {' '.join(sample['tokens'][:30])}...")

    print("\nDone!")
    print("\nNext steps:")
    print("  1. Train causal LM on corpus_tokens.jsonl")
    print("  2. Generate token sequences with transformer")
    print("  3. Decode tokens back to MIDI using the codec")


if __name__ == '__main__':
    main()
