#!/usr/bin/env python3
"""
Compositional Tokenizer V2 for Hierarchical Music Patterns
==========================================================

Uses JSON hierarchy (correct) instead of NPZ (has self-reference bugs).

Token Vocabulary:
- INTERVAL_-54 ... INTERVAL_+54: Interval steps (109 tokens for full range)
- COMPOSE: Merge previous two into parent (1 token)
- GM_0 ... GM_127, GM_128: Instrument context (129 tokens)
- BOS, EOS, PAD: Special tokens (3 tokens)

Encoding:
- Post-order tree traversal: left child, connector, right child, COMPOSE
- Each hierarchical pattern = recursive composition of intervals

Example:
    Tree: R5000 = R234(+7)R567
    Where R234 = leaf(+3), R567 = leaf(-2)
    Tokens: [INTERVAL_+3, CONN_+7, INTERVAL_-2, COMPOSE]
"""

import orjson
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import sys


class CompositionalTokenizerV2:
    """Tokenizer that uses JSON hierarchy (no self-reference bugs)."""

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.checkpoint_path = Path(checkpoint_path)
        self.verbose = verbose

        # JSON patterns
        self.patterns = None

        # Track unique intervals found
        self.unique_intervals = set()

        # Token mappings
        self.vocab = {}
        self.id_to_token = {}
        self.vocab_size = 0

    def load_patterns(self):
        """Load patterns JSON."""
        if self.verbose:
            print("Loading patterns JSON with orjson...")

        patterns_file = self.checkpoint_path / 'checkpoint_v55_pure_contour_1000files_patterns.json'
        with open(patterns_file, 'rb') as f:
            self.patterns = orjson.loads(f.read())

        if self.verbose:
            print(f"  Loaded {len(self.patterns)} patterns")

        # Analyze patterns
        hierarchical = sum(1 for p in self.patterns.values() if p.get('is_hierarchical'))
        leaves = len(self.patterns) - hierarchical

        if self.verbose:
            print(f"  Hierarchical: {hierarchical}, Leaves: {leaves}")

        # Collect all unique intervals
        for data in self.patterns.values():
            for interval in data.get('pitch_intervals', []):
                self.unique_intervals.add(interval)

        if self.verbose:
            print(f"  Unique intervals: {len(self.unique_intervals)}")
            print(f"  Interval range: {min(self.unique_intervals)} to {max(self.unique_intervals)}")

    def _build_vocab(self):
        """Build token vocabulary based on observed intervals."""
        token_id = 0

        # Special tokens
        self.vocab['PAD'] = token_id; token_id += 1
        self.vocab['BOS'] = token_id; token_id += 1
        self.vocab['EOS'] = token_id; token_id += 1

        # Interval tokens for ALL observed intervals
        min_interval = min(self.unique_intervals)
        max_interval = max(self.unique_intervals)

        for interval in range(min_interval, max_interval + 1):
            token_name = f'INTERVAL_{interval:+d}'
            self.vocab[token_name] = token_id
            token_id += 1

        # COMPOSE token
        self.vocab['COMPOSE'] = token_id; token_id += 1

        # GM program tokens (0-128, where 128 = drums)
        for gm in range(129):
            token_name = f'GM_{gm}'
            self.vocab[token_name] = token_id
            token_id += 1

        self.vocab_size = token_id
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        if self.verbose:
            print(f"\n  Vocabulary size: {self.vocab_size}")

    def bucket_interval(self, interval: int) -> int:
        """Get token for interval (with clamping to vocab range)."""
        min_interval = min(self.unique_intervals)
        max_interval = max(self.unique_intervals)
        clamped = max(min_interval, min(max_interval, interval))
        return self.vocab[f'INTERVAL_{clamped:+d}']

    def is_terminal_ref(self, child_name) -> bool:
        """Check if a child reference is a terminal (not a pattern).

        Terminals have rule_id < 129 (first rule is n_terminals + 1 = 128 + 1 = 129).
        They represent single notes with no interval content.
        """
        if child_name is None:
            return True
        if isinstance(child_name, int):
            return child_name < 129
        if not isinstance(child_name, str):
            return True
        if not child_name:
            return True
        parts = child_name.split('_')
        if len(parts) != 2:
            return True
        try:
            rule_id = int(parts[1])
            return rule_id < 129  # Terminals are 0-128
        except ValueError:
            return True

    def encode_pattern(self, pattern_name: str, depth: int = 0, memo: Dict = None) -> List[int]:
        """Encode a pattern as tokens using post-order traversal.

        Post-order: encode left child, then connector interval, then right child, then COMPOSE

        Key insight: Children can be:
        - Terminals (rule_id < 129): Single notes, no interval contribution
        - Patterns (rule_id >= 129): Have interval structure
        """
        if memo is None:
            memo = {}

        if pattern_name in memo:
            return memo[pattern_name]

        if depth > 50:  # Prevent deep recursion
            return []

        if pattern_name not in self.patterns:
            return []

        data = self.patterns[pattern_name]

        # Leaf pattern (2-note, single interval)
        if not data.get('is_hierarchical'):
            intervals = data.get('pitch_intervals', [])
            if len(intervals) == 1:
                tokens = [self.bucket_interval(intervals[0])]
                memo[pattern_name] = tokens
                return tokens
            elif len(intervals) > 1:
                # Multi-interval leaf - chain with COMPOSE
                tokens = [self.bucket_interval(intervals[0])]
                for interval in intervals[1:]:
                    tokens.append(self.bucket_interval(interval))
                    tokens.append(self.vocab['COMPOSE'])
                memo[pattern_name] = tokens
                return tokens
            else:
                return []

        # Hierarchical pattern
        left_child = data.get('left_child', '')
        right_child = data.get('right_child', '')
        connector = data.get('connector_interval', 0)

        # Check if children are terminals or patterns
        left_is_terminal = self.is_terminal_ref(left_child)
        right_is_terminal = self.is_terminal_ref(right_child)

        # Encode children recursively (terminals return empty)
        if left_is_terminal:
            left_tokens = []  # Terminal = single note, no interval
        else:
            left_tokens = self.encode_pattern(left_child, depth + 1, memo)

        if right_is_terminal:
            right_tokens = []  # Terminal = single note, no interval
        else:
            right_tokens = self.encode_pattern(right_child, depth + 1, memo)

        # Build token sequence:
        # - If left is terminal: just connector + right + COMPOSE
        # - If right is terminal: left + connector + COMPOSE
        # - If both are patterns: left + connector + right + COMPOSE
        # - If both terminals: just the connector interval

        if left_is_terminal and right_is_terminal:
            # Base case: two terminals joined by connector = single interval
            tokens = [self.bucket_interval(connector)]
        elif left_is_terminal:
            # Left is terminal, right is pattern
            tokens = [self.bucket_interval(connector)] + right_tokens + [self.vocab['COMPOSE']]
        elif right_is_terminal:
            # Left is pattern, right is terminal
            tokens = left_tokens + [self.bucket_interval(connector), self.vocab['COMPOSE']]
        else:
            # Both are patterns
            tokens = left_tokens + [self.bucket_interval(connector)] + right_tokens + [self.vocab['COMPOSE']]

        memo[pattern_name] = tokens
        return tokens

    def load_and_tokenize(self, output_dir: str):
        """Load patterns and tokenize."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load patterns
        self.load_patterns()

        # Build vocab
        self._build_vocab()

        if self.verbose:
            print("\nEncoding patterns from JSON hierarchy...")

        # Group patterns by rule_id (shared structure across GM programs)
        by_rule_id = defaultdict(list)
        for name in self.patterns.keys():
            parts = name.split('_')
            if len(parts) == 2:
                rule_id = int(parts[1])
                by_rule_id[rule_id].append(name)

        if self.verbose:
            print(f"  Unique rule_ids: {len(by_rule_id)}")

        # Encode each unique rule structure (use first GM variant)
        all_sequences = []
        rule_sequences = {}
        memo = {}

        for rule_id in sorted(by_rule_id.keys()):
            # Take first variant (all have same structure)
            pattern_name = by_rule_id[rule_id][0]

            # Skip leaves for rule sequences (they're just single intervals)
            data = self.patterns[pattern_name]
            if not data.get('is_hierarchical'):
                continue

            tokens = self.encode_pattern(pattern_name, memo=memo)
            if tokens and len(tokens) > 1:  # Skip trivial encodings
                rule_sequences[rule_id] = tokens
                all_sequences.append(tokens)

        if self.verbose:
            print(f"  Encoded {len(rule_sequences)} hierarchical patterns")

        # Statistics
        if all_sequences:
            seq_lengths = [len(s) for s in all_sequences]
            total_tokens = sum(seq_lengths)

            if self.verbose:
                print(f"\n=== STATISTICS ===")
                print(f"  Total sequences: {len(all_sequences)}")
                print(f"  Total tokens: {total_tokens:,}")
                print(f"  Avg sequence length: {np.mean(seq_lengths):.1f}")
                print(f"  Max sequence length: {max(seq_lengths)}")
                print(f"  Min sequence length: {min(seq_lengths)}")
        else:
            total_tokens = 0
            seq_lengths = []

        # Save vocabulary
        vocab_file = output_path / 'vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
                'n_intervals': len(self.unique_intervals),
                'interval_range': [min(self.unique_intervals), max(self.unique_intervals)],
            }, f, indent=2)

        if self.verbose:
            print(f"\n  Saved vocabulary to {vocab_file}")

        # Save rule sequences
        sequences_file = output_path / 'rule_sequences.pt'
        torch.save({
            'sequences': all_sequences,
            'rule_to_sequence': rule_sequences,
        }, sequences_file)

        if self.verbose:
            print(f"  Saved sequences to {sequences_file}")

        # Save statistics
        stats = {
            'n_sequences': len(all_sequences),
            'total_tokens': total_tokens,
            'vocab_size': self.vocab_size,
            'avg_length': float(np.mean(seq_lengths)) if seq_lengths else 0,
            'max_length': int(max(seq_lengths)) if seq_lengths else 0,
            'n_unique_intervals': len(self.unique_intervals),
            'interval_range': [int(min(self.unique_intervals)), int(max(self.unique_intervals))],
        }
        stats_file = output_path / 'stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def decode_tokens(self, tokens: List[int]) -> dict:
        """Decode token sequence back to tree structure (for verification)."""
        stack = []
        result = {'trees': [], 'gm': None}

        for tok in tokens:
            tok_name = self.id_to_token.get(tok, 'UNK')

            if tok_name == 'BOS':
                continue
            elif tok_name == 'EOS':
                break
            elif tok_name.startswith('GM_'):
                result['gm'] = int(tok_name.split('_')[1])
            elif tok_name.startswith('INTERVAL_'):
                interval = int(tok_name.split('_')[1].replace('+', ''))
                stack.append({'type': 'interval', 'value': interval})
            elif tok_name == 'COMPOSE':
                if len(stack) >= 3:
                    right = stack.pop()
                    conn = stack.pop()
                    left = stack.pop()
                    composed = {
                        'type': 'composed',
                        'left': left,
                        'connector': conn.get('value', 0) if isinstance(conn, dict) else conn,
                        'right': right
                    }
                    stack.append(composed)

        result['trees'] = stack
        return result

    def verify_encoding(self, pattern_name: str) -> bool:
        """Verify that encoding produces correct intervals.

        Since our encoding produces flat interval sequences (with COMPOSE markers),
        we verify by extracting intervals from tokens and comparing to original.
        """
        if pattern_name not in self.patterns:
            return False

        data = self.patterns[pattern_name]
        original_intervals = data.get('pitch_intervals', [])

        tokens = self.encode_pattern(pattern_name)

        # Extract intervals from token sequence (ignore COMPOSE tokens)
        reconstructed = []
        for tok in tokens:
            tok_name = self.id_to_token.get(tok, '')
            if tok_name.startswith('INTERVAL_'):
                # Parse interval value
                val_str = tok_name.split('_')[1]
                interval = int(val_str.replace('+', ''))
                reconstructed.append(interval)

        return reconstructed == original_intervals


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compositional tokenizer V2 using JSON hierarchy')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based',
                        help='Directory containing checkpoint files')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/compositional_tokens_v2',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Verify encoding correctness')

    args = parser.parse_args()

    tokenizer = CompositionalTokenizerV2(args.checkpoint_dir, verbose=not args.quiet)
    stats = tokenizer.load_and_tokenize(args.output_dir)

    print(f"\nTokenization complete!")
    print(f"  Vocab size: {stats['vocab_size']}")
    print(f"  Sequences: {stats['n_sequences']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")

    if args.verify:
        print("\nVerifying encodings...")
        correct = 0
        total = 0
        for name in list(tokenizer.patterns.keys())[:100]:
            if tokenizer.patterns[name].get('is_hierarchical'):
                if tokenizer.verify_encoding(name):
                    correct += 1
                total += 1
        print(f"  Verified {correct}/{total} ({100*correct/total:.1f}%)")


if __name__ == '__main__':
    main()
