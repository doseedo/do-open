#!/usr/bin/env python3
"""
Compositional Tokenizer for Hierarchical Music Patterns
========================================================

Converts the v55 checkpoint hierarchy into compositional token sequences.

Token Vocabulary (~105 tokens):
- LEAF_0 ... LEAF_44: 2-note building blocks (45 tokens)
- CONN_-12 ... CONN_+12: Connector intervals bucketed (25 tokens)
- COMPOSE: Merge previous two into parent (1 token)
- TRACK_0 ... TRACK_14: GM program context (15 tokens)
- BEAT_0 ... BEAT_15: Time advance buckets (16 tokens)
- BOS, EOS, PAD: Special tokens (3 tokens)

Encoding:
- Post-order tree traversal: children first, then COMPOSE
- Track context before each track's patterns
- Beat advances between patterns

Example:
    Tree: R5000 = R234(+7)R567
    Tokens: [LEAF_3, CONN_-2, LEAF_12, COMPOSE, CONN_+7, LEAF_7, COMPOSE]
"""

import numpy as np
import orjson
import json
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

# Constants
TICKS_PER_BEAT = 480
BEAT_BUCKETS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
CONNECTOR_BUCKETS = list(range(-12, 13))  # -12 to +12

# GM program mapping (top 15 by usage)
TOP_15_GMS = [128, 25, 24, 27, 0, 71, 56, 32, 33, 67, 57, 65, 66, 60, 73]
GM_TO_TRACK_ID = {gm: i for i, gm in enumerate(TOP_15_GMS)}


class CompositionalTokenizer:
    """Tokenizer that encodes hierarchical patterns as compositional sequences."""

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.checkpoint_path = Path(checkpoint_path)
        self.verbose = verbose

        # Will be populated by load_checkpoint
        self.rule_contours = None
        self.rule_children = None
        self.rule_counts = None
        self.n_terminals = None
        self.n_rules = None

        # Token mappings
        self.leaf_to_token = {}
        self.token_to_leaf = {}
        self.vocab = {}
        self.vocab_size = 0

    def load_checkpoint(self):
        """Load NPZ checkpoint and build token mappings."""
        if self.verbose:
            print("Loading checkpoint...")

        ckpt_file = self.checkpoint_path / 'checkpoint_v55_pure_contour_1000files.npz'
        ckpt = np.load(ckpt_file, allow_pickle=True)

        self.rule_contours = ckpt['rule_contours']
        self.rule_children = ckpt['rule_children']
        self.rule_counts = ckpt['rule_counts']
        self.n_terminals = int(ckpt['n_terminals'][0])
        self.n_rules = int(ckpt['n_rules'][0])

        if self.verbose:
            print(f"  Loaded {self.n_rules} rules, {self.n_terminals} terminals")

        self._build_vocab()

    def _build_vocab(self):
        """Build token vocabulary."""
        token_id = 0

        # Special tokens
        self.vocab['PAD'] = token_id; token_id += 1
        self.vocab['BOS'] = token_id; token_id += 1
        self.vocab['EOS'] = token_id; token_id += 1

        # Leaf tokens (45 leaves)
        leaf_rules = []
        for rule_idx in range(self.n_rules):
            if self.rule_children[rule_idx, 0] < 0:  # No children = leaf
                rule_id = self.n_terminals + rule_idx
                leaf_rules.append(rule_id)

        for i, rule_id in enumerate(sorted(leaf_rules)):
            token_name = f'LEAF_{i}'
            self.vocab[token_name] = token_id
            self.leaf_to_token[rule_id] = token_id
            self.token_to_leaf[token_id] = rule_id
            token_id += 1

        # Connector tokens (-12 to +12, bucket larger)
        for conn in CONNECTOR_BUCKETS:
            token_name = f'CONN_{conn:+d}'
            self.vocab[token_name] = token_id
            token_id += 1

        # COMPOSE token
        self.vocab['COMPOSE'] = token_id; token_id += 1

        # Track tokens (15 GM programs)
        for i in range(15):
            token_name = f'TRACK_{i}'
            self.vocab[token_name] = token_id
            token_id += 1

        # Beat advance tokens (16 buckets)
        for i in range(len(BEAT_BUCKETS)):
            token_name = f'BEAT_{i}'
            self.vocab[token_name] = token_id
            token_id += 1

        self.vocab_size = token_id

        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        if self.verbose:
            print(f"  Vocabulary size: {self.vocab_size}")
            print(f"  Leaf tokens: {len(self.leaf_to_token)}")

    def bucket_connector(self, interval: int) -> int:
        """Bucket connector interval to -12..+12 range."""
        if interval < -12:
            return -12
        elif interval > 12:
            return 12
        return interval

    def bucket_beat(self, beats: float) -> int:
        """Bucket beat delta to predefined buckets."""
        for i, b in enumerate(BEAT_BUCKETS):
            if beats <= b:
                return i
        return len(BEAT_BUCKETS) - 1

    def is_leaf(self, rule_id: int) -> bool:
        """Check if rule is a leaf (no children)."""
        if rule_id < self.n_terminals:
            return True  # Terminal
        rule_idx = rule_id - self.n_terminals
        if rule_idx >= self.n_rules:
            return True
        return self.rule_children[rule_idx, 0] < 0

    def encode_tree(self, rule_id: int, depth: int = 0) -> List[int]:
        """Encode a rule tree as tokens using post-order traversal.

        Post-order: left children, connector, right children, COMPOSE
        """
        if depth > 100:  # Prevent infinite recursion
            return []

        # Leaf node
        if self.is_leaf(rule_id):
            if rule_id in self.leaf_to_token:
                return [self.leaf_to_token[rule_id]]
            else:
                # Unknown leaf - skip
                return []

        rule_idx = rule_id - self.n_terminals
        left_child = int(self.rule_children[rule_idx, 0])
        right_child = int(self.rule_children[rule_idx, 1])
        connector = int(self.rule_contours[rule_idx, 0])

        # Encode children
        left_tokens = self.encode_tree(left_child, depth + 1)
        right_tokens = self.encode_tree(right_child, depth + 1)

        if not left_tokens or not right_tokens:
            return []  # Skip if children couldn't be encoded

        # Connector token
        bucketed = self.bucket_connector(connector)
        conn_token = self.vocab[f'CONN_{bucketed:+d}']

        # Post-order: left, connector, right, COMPOSE
        return left_tokens + [conn_token] + right_tokens + [self.vocab['COMPOSE']]

    def encode_piece(self, piece_occs: List[Dict], piece_id: str) -> List[int]:
        """Encode a piece's occurrences as token sequence.

        Args:
            piece_occs: List of occurrence dicts with track, onset, rule_id
            piece_id: Piece identifier for debugging

        Returns:
            List of token IDs
        """
        if not piece_occs:
            return []

        tokens = [self.vocab['BOS']]

        # Group by track
        by_track = defaultdict(list)
        for occ in piece_occs:
            track = occ.get('track_id', occ.get('track', 0))
            by_track[track].append(occ)

        for track_id in sorted(by_track.keys()):
            track_occs = by_track[track_id]

            # Get GM program for this track
            gm = track_occs[0].get('gm_program', 0)
            track_token_id = GM_TO_TRACK_ID.get(gm, 0)
            tokens.append(self.vocab[f'TRACK_{track_token_id}'])

            # Sort by onset time
            track_occs.sort(key=lambda x: x.get('onset_time', 0))

            prev_beat = 0
            for occ in track_occs:
                onset = occ.get('onset_time', 0)
                current_beat = onset / TICKS_PER_BEAT

                # Beat advance
                delta_beats = current_beat - prev_beat
                if delta_beats > 0:
                    beat_bucket = self.bucket_beat(delta_beats)
                    tokens.append(self.vocab[f'BEAT_{beat_bucket}'])

                prev_beat = current_beat

                # Encode the pattern tree
                # For now, encode as flat pattern since we need rule_id mapping
                # TODO: Map JSON pattern to NPZ rule_id and encode tree

        tokens.append(self.vocab['EOS'])
        return tokens

    def load_and_tokenize_corpus(self, output_dir: str):
        """Load patterns and tokenize entire corpus."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        self.load_checkpoint()

        # Load patterns JSON
        if self.verbose:
            print("Loading patterns JSON...")

        patterns_file = self.checkpoint_path / 'checkpoint_v55_pure_contour_1000files_patterns.json'
        with open(patterns_file, 'rb') as f:
            patterns = orjson.loads(f.read())

        if self.verbose:
            print(f"  Loaded {len(patterns)} patterns")

        # Group occurrences by piece
        piece_occs = defaultdict(list)
        for pat_name, pat_data in patterns.items():
            for occ in pat_data.get('occurrences', []):
                piece_occs[occ['piece_id']].append({
                    **occ,
                    'pattern_name': pat_name,
                    'canonical_pitches': pat_data['canonical_pitches'],
                    'pitch_intervals': pat_data['pitch_intervals'],
                })

        if self.verbose:
            print(f"  Found {len(piece_occs)} pieces")

        # For compositional encoding, we need to map JSON patterns to NPZ rule trees
        # This requires understanding which JSON pattern corresponds to which rule_id
        # For now, let's encode the STRUCTURE from NPZ directly

        # Encode each rule tree and create training data
        if self.verbose:
            print("\nEncoding rule trees...")

        all_sequences = []
        rule_sequences = {}

        # Encode each non-leaf rule as a tree
        for rule_idx in range(min(self.n_rules, 5000)):  # Limit for now
            rule_id = self.n_terminals + rule_idx
            if not self.is_leaf(rule_id):
                tokens = self.encode_tree(rule_id)
                if tokens:
                    rule_sequences[rule_id] = tokens
                    all_sequences.append(tokens)

        if self.verbose:
            print(f"  Encoded {len(rule_sequences)} rule trees")

        # Statistics
        seq_lengths = [len(s) for s in all_sequences]
        total_tokens = sum(seq_lengths)

        if self.verbose:
            print(f"\n=== STATISTICS ===")
            print(f"  Total sequences: {len(all_sequences)}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Avg sequence length: {np.mean(seq_lengths):.1f}")
            print(f"  Max sequence length: {max(seq_lengths)}")

        # Save vocabulary
        vocab_file = output_path / 'vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
                'vocab_size': self.vocab_size,
                'leaf_to_token': {str(k): v for k, v in self.leaf_to_token.items()},
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
            'avg_length': float(np.mean(seq_lengths)),
            'max_length': int(max(seq_lengths)),
            'n_leaves': len(self.leaf_to_token),
        }
        stats_file = output_path / 'stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def decode_tokens(self, tokens: List[int]) -> dict:
        """Decode token sequence back to tree structure (for verification)."""
        stack = []
        result = {'trees': [], 'track': None, 'beats': []}

        for tok in tokens:
            tok_name = self.id_to_token.get(tok, 'UNK')

            if tok_name == 'BOS':
                continue
            elif tok_name == 'EOS':
                break
            elif tok_name.startswith('TRACK_'):
                result['track'] = int(tok_name.split('_')[1])
            elif tok_name.startswith('BEAT_'):
                beat_idx = int(tok_name.split('_')[1])
                result['beats'].append(BEAT_BUCKETS[beat_idx] if beat_idx < len(BEAT_BUCKETS) else beat_idx)
            elif tok_name.startswith('LEAF_'):
                leaf_id = int(tok_name.split('_')[1])
                stack.append({'type': 'leaf', 'id': leaf_id})
            elif tok_name.startswith('CONN_'):
                conn = int(tok_name.split('_')[1].replace('+', ''))
                stack.append({'type': 'conn', 'value': conn})
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compositional tokenizer for hierarchical patterns')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based',
                        help='Directory containing checkpoint files')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/compositional_tokens',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    tokenizer = CompositionalTokenizer(args.checkpoint_dir, verbose=not args.quiet)
    stats = tokenizer.load_and_tokenize_corpus(args.output_dir)

    print(f"\nTokenization complete!")
    print(f"  Vocab size: {stats['vocab_size']}")
    print(f"  Sequences: {stats['n_sequences']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")


if __name__ == '__main__':
    main()
