#!/usr/bin/env python3
"""
Tokenization Serialization Script
==================================

Converts the v55 checkpoint into token sequences for transformer training.

Token Format:
    (pattern_id, pitch_offset, delta_8ths)
    
    - pattern_id: "GM65_1045" style ID (instrument-specific)
    - pitch_offset: 0-11 (transposition from canonical)
    - delta_8ths: quantized time delta in 8th notes (0, 1, 2, 4, 8...)

Quantization:
    - 480 ticks/beat → 240 ticks/8th note
    - ±240 tick tolerance for swing quantization
    - Snaps to nearest 8th note grid

Filtering:
    - Only pieces that contain instruments from top 15 GM programs
    - Non-overlapping derivations from track_derives
"""

import os
import sys
import json
import orjson
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Constants
TICKS_PER_BEAT = 480
TICKS_PER_8TH = 240  # 8th note = 0.5 beats

# Delta time buckets (in 8th notes)
DELTA_BUCKETS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]

def quantize_to_8th(ticks: int) -> int:
    """Quantize ticks to nearest 8th note."""
    return round(ticks / TICKS_PER_8TH)

def bucket_delta(delta_8ths: int) -> int:
    """Bucket delta time to predefined buckets."""
    for i, b in enumerate(DELTA_BUCKETS):
        if delta_8ths <= b:
            return i
    return len(DELTA_BUCKETS) - 1

def get_pitch_offset(first_pitch: int, canonical_first: int) -> int:
    """Get pitch offset (transposition) mod 12."""
    return (first_pitch - canonical_first) % 12


class TokenSerializer:
    def __init__(self, checkpoint_dir: str, verbose: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose
        self.patterns = None
        self.top_gms = None
        
    def load_data(self):
        """Load patterns and identify top 15 GM programs."""
        if self.verbose:
            print("Loading patterns...")
        
        patterns_file = self.checkpoint_dir / 'checkpoint_v55_pure_contour_1000files_patterns.json'
        with open(patterns_file, 'rb') as f:
            self.patterns = orjson.loads(f.read())
        
        if self.verbose:
            print(f"  Loaded {len(self.patterns):,} patterns")
        
        # Identify top 15 GM programs by note coverage
        gm_notes = defaultdict(int)
        for pat_name, pat_data in self.patterns.items():
            n_notes = len(pat_data['canonical_pitches'])
            for occ in pat_data.get('occurrences', []):
                gm_notes[occ['gm_program']] += n_notes
        
        sorted_gms = sorted(gm_notes.items(), key=lambda x: x[1], reverse=True)
        self.top_gms = set(gm for gm, _ in sorted_gms[:15])
        
        if self.verbose:
            print(f"  Top 15 GM programs: {sorted(self.top_gms)}")
            print(f"  Total notes in top 15: {sum(n for gm, n in sorted_gms[:15]):,}")
    
    def get_valid_pieces(self) -> List[str]:
        """Get pieces that only use top 15 instruments."""
        piece_gms = defaultdict(set)
        
        for pat_name, pat_data in self.patterns.items():
            for occ in pat_data.get('occurrences', []):
                piece_gms[occ['piece_id']].add(occ['gm_program'])
        
        valid = []
        for piece, gms in piece_gms.items():
            if gms.issubset(self.top_gms):
                valid.append(piece)
        
        if self.verbose:
            print(f"\n  Pieces with only top 15 instruments: {len(valid)} / {len(piece_gms)}")
        
        return valid
    
    def extract_derivation(self, piece_id: str) -> List[Dict]:
        """Extract non-overlapping derivation for a piece."""
        # Collect all occurrences
        all_occs = []
        for pat_name, pat_data in self.patterns.items():
            for occ in pat_data.get('occurrences', []):
                if occ['piece_id'] == piece_id:
                    all_occs.append({
                        'pattern_id': pat_name,
                        'track_id': occ['track_id'],
                        'gm_program': occ['gm_program'],
                        'position': occ['position'],
                        'last_position': occ['last_position'],
                        'onset_time': occ['onset_time'],
                        'first_pitch': occ['first_pitch'],
                        'canonical_first': pat_data['canonical_pitches'][0],
                        'n_notes': len(pat_data['canonical_pitches']),
                    })
        
        # Group by track
        by_track = defaultdict(list)
        for occ in all_occs:
            by_track[occ['track_id']].append(occ)
        
        # Non-overlapping selection per track (greedy, longest first)
        derivation = []
        for track_id, occs in by_track.items():
            # Group by start position
            pos_to_occs = defaultdict(list)
            for o in occs:
                pos_to_occs[o['position']].append(o)
            
            selected = []
            current_end = -1
            for pos in sorted(pos_to_occs.keys()):
                if pos <= current_end:
                    continue
                # Take longest pattern at this position
                longest = max(pos_to_occs[pos], key=lambda x: x['n_notes'])
                selected.append(longest)
                current_end = longest['last_position']
            
            derivation.extend(selected)
        
        return derivation
    
    def tokenize_piece(self, piece_id: str) -> List[Tuple[str, int, int]]:
        """Convert piece to token sequence.
        
        Returns:
            List of (pattern_id, pitch_offset, delta_bucket) tuples
        """
        derivation = self.extract_derivation(piece_id)
        
        if not derivation:
            return []
        
        # Sort by onset time, then track
        derivation.sort(key=lambda x: (x['onset_time'], x['track_id']))
        
        # Convert to tokens with delta times
        tokens = []
        prev_onset = 0
        
        for occ in derivation:
            pattern_id = occ['pattern_id']
            pitch_offset = get_pitch_offset(occ['first_pitch'], occ['canonical_first'])
            
            # Delta time in 8th notes
            delta_ticks = occ['onset_time'] - prev_onset
            delta_8ths = quantize_to_8th(delta_ticks)
            delta_bucket = bucket_delta(delta_8ths)
            
            tokens.append((pattern_id, pitch_offset, delta_bucket))
            prev_onset = occ['onset_time']
        
        return tokens
    
    def build_vocabulary(self, token_sequences: List[List[Tuple]]) -> Dict:
        """Build pattern vocabulary from token sequences."""
        pattern_counts = defaultdict(int)
        
        for seq in token_sequences:
            for pattern_id, pitch_offset, delta_bucket in seq:
                pattern_counts[pattern_id] += 1
        
        # Sort by frequency
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        vocab = {
            'pattern_to_id': {pat: i for i, (pat, _) in enumerate(sorted_patterns)},
            'id_to_pattern': {i: pat for i, (pat, _) in enumerate(sorted_patterns)},
            'pattern_counts': dict(sorted_patterns),
            'n_patterns': len(sorted_patterns),
            'n_pitch_offsets': 12,
            'n_delta_buckets': len(DELTA_BUCKETS),
        }
        
        return vocab
    
    def serialize_corpus(self, output_dir: str, top_k_patterns: Optional[int] = None):
        """Serialize entire corpus to token sequences."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Get valid pieces
        valid_pieces = self.get_valid_pieces()
        
        if self.verbose:
            print(f"\nTokenizing {len(valid_pieces)} pieces...")
        
        # Tokenize all pieces
        token_sequences = {}
        total_tokens = 0
        
        for i, piece_id in enumerate(valid_pieces):
            tokens = self.tokenize_piece(piece_id)
            if tokens:
                token_sequences[piece_id] = tokens
                total_tokens += len(tokens)
            
            if self.verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(valid_pieces)} pieces...")
        
        if self.verbose:
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Pieces with tokens: {len(token_sequences)}")
        
        # Build vocabulary
        vocab = self.build_vocabulary(list(token_sequences.values()))
        
        if self.verbose:
            print(f"\n  Vocabulary size: {vocab['n_patterns']:,} patterns")
            print(f"  Top 10 patterns by frequency:")
            for pat, count in list(vocab['pattern_counts'].items())[:10]:
                print(f"    {pat}: {count:,}")
        
        # Filter to top_k patterns if specified
        if top_k_patterns and top_k_patterns < vocab['n_patterns']:
            if self.verbose:
                print(f"\n  Filtering to top {top_k_patterns} patterns...")
            
            top_patterns = set(list(vocab['pattern_to_id'].keys())[:top_k_patterns])
            
            # Filter sequences
            filtered_sequences = {}
            filtered_tokens = 0
            for piece_id, seq in token_sequences.items():
                filtered = [(p, po, d) for p, po, d in seq if p in top_patterns]
                if filtered:
                    filtered_sequences[piece_id] = filtered
                    filtered_tokens += len(filtered)
            
            token_sequences = filtered_sequences
            
            # Rebuild vocab
            vocab = self.build_vocabulary(list(token_sequences.values()))
            
            if self.verbose:
                print(f"  Filtered tokens: {filtered_tokens:,}")
                print(f"  Filtered pieces: {len(token_sequences)}")
        
        # Compute statistics
        seq_lengths = [len(seq) for seq in token_sequences.values()]
        stats = {
            'n_pieces': len(token_sequences),
            'total_tokens': sum(seq_lengths),
            'min_length': min(seq_lengths) if seq_lengths else 0,
            'max_length': max(seq_lengths) if seq_lengths else 0,
            'mean_length': np.mean(seq_lengths) if seq_lengths else 0,
            'median_length': np.median(seq_lengths) if seq_lengths else 0,
        }
        
        if self.verbose:
            print(f"\n=== CORPUS STATISTICS ===")
            print(f"  Pieces: {stats['n_pieces']}")
            print(f"  Total tokens: {stats['total_tokens']:,}")
            print(f"  Sequence lengths: min={stats['min_length']}, max={stats['max_length']}, "
                  f"mean={stats['mean_length']:.1f}, median={stats['median_length']:.1f}")
        
        # Save outputs
        if self.verbose:
            print(f"\nSaving to {output_path}...")
        
        # Save vocabulary
        with open(output_path / 'vocabulary.json', 'w') as f:
            json.dump(vocab, f, indent=2)
        
        # Save token sequences (as numpy arrays for efficiency)
        sequences_data = {
            'piece_ids': list(token_sequences.keys()),
            'sequences': [seq for seq in token_sequences.values()],
        }
        np.savez_compressed(output_path / 'token_sequences.npz', **{
            'piece_ids': np.array(sequences_data['piece_ids'], dtype=object),
            'n_sequences': len(sequences_data['sequences']),
        })
        
        # Save sequences as JSON for readability
        with open(output_path / 'token_sequences.json', 'w') as f:
            json.dump(sequences_data, f)
        
        # Save statistics
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        if self.verbose:
            print("  Done!")
        
        return vocab, token_sequences, stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Serialize corpus to tokens')
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based',
                        help='Directory containing checkpoint files')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/tokenized',
                        help='Output directory for serialized tokens')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Filter to top K patterns by frequency')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    serializer = TokenSerializer(args.checkpoint_dir, verbose=not args.quiet)
    vocab, sequences, stats = serializer.serialize_corpus(args.output_dir, args.top_k)
    
    print(f"\nSerialization complete!")
    print(f"  Output: {args.output_dir}")
    print(f"  Vocabulary: {vocab['n_patterns']} patterns")
    print(f"  Sequences: {stats['n_pieces']} pieces, {stats['total_tokens']} tokens")


if __name__ == '__main__':
    main()
