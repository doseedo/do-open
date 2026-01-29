#!/usr/bin/env python3
"""
Piece-Level Tokenizer for Music Generation
==========================================

Treats discovered patterns as atomic tokens (fixed vocabulary).
The model learns: what patterns follow what patterns in real music.

Philosophy: "Discovery not Prescription"
- Patterns are ALREADY discovered via Re-Pair compression
- This tokenizer uses them as vocabulary, not as things to generate

Token Types:
- PATTERN_{gm}_{rule_id}: Pattern occurrence (e.g., PATTERN_GM65_234)
- TRACK_{gm}: Instrument/track marker
- BEAT_{n}: Time position marker (quantized)
- BOS, EOS, PAD: Special tokens
"""

import json
import orjson
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
from tqdm import tqdm


class PieceLevelTokenizer:
    """
    Tokenizer that treats patterns as atomic tokens.

    Extracts sequences from piece occurrences data where each
    pattern usage becomes a single token in the sequence.
    """

    def __init__(
        self,
        patterns_path: str,
        beat_quantization: int = 4,  # Beats per bar (or quantization level)
        max_beat_token: int = 256,   # Maximum beat position token
        ticks_per_beat: int = 480,   # Standard MIDI resolution
    ):
        """
        Args:
            patterns_path: Path to patterns JSON (checkpoint_v55..._patterns.json)
            beat_quantization: How to quantize beat positions
            max_beat_token: Maximum number of beat position tokens
            ticks_per_beat: MIDI ticks per beat (for time conversion)
        """
        self.beat_quantization = beat_quantization
        self.max_beat_token = max_beat_token
        self.ticks_per_beat = ticks_per_beat

        print("Loading patterns data...")
        with open(patterns_path, 'rb') as f:
            self.patterns = orjson.loads(f.read())

        print(f"Loaded {len(self.patterns)} patterns")

        # Build vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from all patterns in JSON."""
        self.vocab = {}
        token_id = 0

        # Special tokens
        for special in ['PAD', 'BOS', 'EOS', 'SEP']:
            self.vocab[special] = token_id
            token_id += 1

        # Track tokens (GM instruments 0-128)
        for gm in range(129):
            self.vocab[f'TRACK_{gm}'] = token_id
            token_id += 1

        # Beat position tokens
        for beat in range(self.max_beat_token):
            self.vocab[f'BEAT_{beat}'] = token_id
            token_id += 1

        # Pattern tokens - one per unique pattern in JSON
        # Pattern keys are like "GM128_129", "GM65_234", etc.
        self.pattern_tokens = set()
        for pattern_key in self.patterns.keys():
            token_name = f'PATTERN_{pattern_key}'
            if token_name not in self.vocab:
                self.vocab[token_name] = token_id
                self.pattern_tokens.add(token_name)
                token_id += 1

        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Special token IDs
        self.pad_id = self.vocab['PAD']
        self.bos_id = self.vocab['BOS']
        self.eos_id = self.vocab['EOS']
        self.sep_id = self.vocab['SEP']

        print(f"Vocabulary built:")
        print(f"  Special tokens: 4")
        print(f"  Track tokens: 129")
        print(f"  Beat tokens: {self.max_beat_token}")
        print(f"  Pattern tokens: {len(self.pattern_tokens)}")
        print(f"  Total vocab size: {self.vocab_size}")

    def _quantize_beat(self, beat: float) -> int:
        """Quantize beat position to token index."""
        # Round to nearest quantization unit
        quantized = int(round(beat * self.beat_quantization) / self.beat_quantization)
        return min(quantized, self.max_beat_token - 1)

    def _ticks_to_beat(self, onset_time: int) -> float:
        """Convert onset time (ticks) to beat position."""
        return onset_time / self.ticks_per_beat

    def encode_piece(
        self,
        occurrences: List[Dict],
        include_beats: bool = True,
        include_tracks: bool = True,
    ) -> List[int]:
        """
        Encode a single piece's occurrences into token sequence.

        Args:
            occurrences: List of pattern occurrences with keys:
                - 'pattern_key': Pattern key (e.g., 'GM65_234')
                - 'gm_program': General MIDI instrument (0-128)
                - 'onset_time': Onset time in ticks
            include_beats: Include BEAT tokens
            include_tracks: Include TRACK tokens

        Returns:
            List of token IDs
        """
        if not occurrences:
            return [self.bos_id, self.eos_id]

        # Sort by onset time, then by GM program
        sorted_occs = sorted(occurrences, key=lambda x: (x.get('onset_time', 0), x.get('gm_program', 0)))

        tokens = [self.bos_id]
        current_track = None
        current_beat = None

        for occ in sorted_occs:
            gm = occ.get('gm_program', 0)
            pattern_key = occ.get('pattern_key', '')
            onset_time = occ.get('onset_time', 0)
            beat = self._ticks_to_beat(onset_time)

            # Add track token if changed
            if include_tracks and gm != current_track:
                track_token = f'TRACK_{gm}'
                if track_token in self.vocab:
                    tokens.append(self.vocab[track_token])
                    current_track = gm

            # Add beat token if changed (quantized)
            if include_beats:
                q_beat = self._quantize_beat(beat)
                if q_beat != current_beat:
                    beat_token = f'BEAT_{q_beat}'
                    if beat_token in self.vocab:
                        tokens.append(self.vocab[beat_token])
                        current_beat = q_beat

            # Add pattern token
            pattern_token = f'PATTERN_{pattern_key}'
            if pattern_token in self.vocab:
                tokens.append(self.vocab[pattern_token])
            # Skip unknown patterns (shouldn't happen with proper data)

        tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs back to token names."""
        return [self.id_to_token.get(tid, f'UNK_{tid}') for tid in token_ids]

    def save_vocab(self, output_path: str):
        """Save vocabulary to JSON file."""
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'n_pattern_tokens': len(self.pattern_tokens),
            'n_track_tokens': 129,
            'n_beat_tokens': self.max_beat_token,
            'beat_quantization': self.beat_quantization,
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved vocabulary to {output_path}")


def extract_piece_sequences(
    patterns_path: str,
    output_dir: str,
    include_beats: bool = True,
    include_tracks: bool = True,
    max_beat_token: int = 256,
):
    """
    Extract piece-level sequences from pattern occurrences.

    The patterns JSON contains pattern definitions AND their occurrences
    in each piece. This function extracts those occurrences and tokenizes them.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tokenizer
    tokenizer = PieceLevelTokenizer(
        patterns_path=patterns_path,
        max_beat_token=max_beat_token,
    )

    print("\nExtracting piece sequences from occurrences...")

    # Collect all occurrences by piece
    pieces = defaultdict(list)

    for pattern_key, pattern_data in tqdm(tokenizer.patterns.items(), desc="Processing patterns"):
        if 'occurrences' not in pattern_data:
            continue

        gm_program = pattern_data.get('gm_program', 0)

        for occ in pattern_data['occurrences']:
            piece_id = occ.get('piece_id', 'unknown')
            pieces[piece_id].append({
                'pattern_key': pattern_key,
                'gm_program': occ.get('gm_program', gm_program),
                'onset_time': occ.get('onset_time', 0),
            })

    print(f"\nFound {len(pieces)} pieces")

    # Tokenize each piece
    all_sequences = []
    piece_names = []

    for piece_name, occs in tqdm(pieces.items(), desc="Tokenizing pieces"):
        tokens = tokenizer.encode_piece(
            occs,
            include_beats=include_beats,
            include_tracks=include_tracks,
        )
        all_sequences.append(tokens)
        piece_names.append(piece_name)

    # Calculate statistics
    lengths = [len(s) for s in all_sequences]
    total_tokens = sum(lengths)

    stats = {
        'n_pieces': len(pieces),
        'total_tokens': total_tokens,
        'vocab_size': tokenizer.vocab_size,
        'avg_length': total_tokens / len(pieces) if pieces else 0,
        'max_length': max(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'n_pattern_tokens': len(tokenizer.pattern_tokens),
        'include_beats': include_beats,
        'include_tracks': include_tracks,
    }

    print(f"\nStatistics:")
    print(f"  Pieces: {stats['n_pieces']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Vocab size: {stats['vocab_size']:,}")
    print(f"  Avg length: {stats['avg_length']:.1f}")
    print(f"  Max length: {stats['max_length']}")

    # Save outputs
    tokenizer.save_vocab(output_dir / 'vocab.json')

    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    torch.save({
        'sequences': all_sequences,
        'piece_names': piece_names,
    }, output_dir / 'piece_sequences.pt')

    print(f"\nSaved to {output_dir}")
    return tokenizer, all_sequences, stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create piece-level tokenization')
    parser.add_argument('--patterns', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/checkpoint_v55_pure_contour_1000files_patterns.json',
                        help='Path to patterns JSON')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/piece_level_tokens',
                        help='Output directory')
    parser.add_argument('--no-beats', action='store_true',
                        help='Exclude beat position tokens')
    parser.add_argument('--no-tracks', action='store_true',
                        help='Exclude track tokens')
    parser.add_argument('--max-beat-token', type=int, default=256,
                        help='Maximum beat position token')

    args = parser.parse_args()

    extract_piece_sequences(
        patterns_path=args.patterns,
        output_dir=args.output_dir,
        include_beats=not args.no_beats,
        include_tracks=not args.no_tracks,
        max_beat_token=args.max_beat_token,
    )
