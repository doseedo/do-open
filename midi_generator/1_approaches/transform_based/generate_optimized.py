#!/usr/bin/env python3
"""
OPTIMIZED MIDI GENERATION
=========================
Only loads the patterns actually used by the model (5000 of 143K).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import argparse
import sys
from pathlib import Path
from midiutil import MIDIFile

GM_NAMES = {
    0: 'piano', 24: 'nylon_guitar', 25: 'steel_guitar', 27: 'clean_guitar',
    32: 'acoustic_bass', 33: 'electric_bass', 56: 'trumpet', 57: 'trombone',
    60: 'french_horn', 65: 'alto_sax', 66: 'tenor_sax', 67: 'baritone_sax',
    72: 'flute', 73: 'piccolo', 128: 'drums'
}

# ============================================================================
# MODEL
# ============================================================================

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
        return self.dropout(x + self.pe[:, :x.size(1)])


class PatternIDTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, dropout=0.1, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        pad_mask = (x == 0)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x, mask=mask, src_key_padding_mask=pad_mask)
        return self.fc_out(x)


# ============================================================================
# OPTIMIZED PATTERN LOADER - Only loads needed patterns
# ============================================================================

def load_needed_patterns(patterns_path, pattern_id_to_name):
    """
    Load only the patterns referenced in pattern_id_to_name.
    This is much faster than loading all 143K patterns.
    """
    needed_names = set(pattern_id_to_name.values())
    print(f"Loading {len(needed_names)} needed patterns (of 143K total)...")
    sys.stdout.flush()

    patterns = {}

    # Stream through the file and only parse needed patterns
    import re

    with open(patterns_path, 'rb') as f:
        content = f.read().decode('utf-8')

    # Find each pattern by name
    found = 0
    for name in needed_names:
        # Find this pattern in the file
        pattern = f'"{name}":'
        idx = content.find(pattern)
        if idx == -1:
            continue

        # Find the opening brace
        start = content.find('{', idx)
        if start == -1:
            continue

        # Find matching closing brace
        brace_count = 0
        end = start
        for i, c in enumerate(content[start:start+100000]):  # Patterns shouldn't be >100KB
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break

        if end > start:
            try:
                patterns[name] = json.loads(content[start:end])
                found += 1
                if found % 500 == 0:
                    print(f"  Loaded {found}/{len(needed_names)} patterns...")
                    sys.stdout.flush()
            except:
                pass

    print(f"  Loaded {len(patterns)} patterns")
    sys.stdout.flush()
    return patterns


# ============================================================================
# DECODE
# ============================================================================

def decode_pattern(pattern_name, patterns, transpose=0, time_offset=0.0):
    """Decode a pattern using ALL factored data."""
    if pattern_name not in patterns:
        return []

    pdata = patterns[pattern_name]
    canonical_pitches = pdata.get('canonical_pitches', [60])
    rhythm_ratios = pdata.get('rhythm_ratios', [1.0] * len(canonical_pitches))
    duration_ratios = pdata.get('duration_ratios', [1.0] * len(canonical_pitches))
    velocity_ratios = pdata.get('velocity_ratios', [1.0] * len(canonical_pitches))
    gm_program = pdata.get('gm_program', 0)

    base_duration = 0.5
    notes = []
    current_time = time_offset
    pitches = [p + transpose for p in canonical_pitches]

    for i, pitch in enumerate(pitches):
        dur = duration_ratios[i] * base_duration if i < len(duration_ratios) else base_duration
        vel = int(min(127, max(1, velocity_ratios[i] * 80))) if i < len(velocity_ratios) else 80

        notes.append({
            'pitch': max(21, min(108, pitch)),
            'time': current_time,
            'duration': max(0.1, dur),
            'velocity': vel,
            'gm_program': gm_program
        })

        if i + 1 < len(rhythm_ratios):
            current_time += rhythm_ratios[i] * base_duration
        else:
            current_time += base_duration

    return notes


# ============================================================================
# GENERATION
# ============================================================================

def generate_tokens(model, vocab, device, max_length=256, temperature=0.85, top_k=50, top_p=0.92):
    """Generate a sequence of tokens."""
    model.eval()
    bos_id = vocab.get('BOS', 1)
    eos_id = vocab.get('EOS', 2)
    tokens = [bos_id]

    with torch.no_grad():
        for _ in range(max_length - 1):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1] / temperature

            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, k)[0][-1]
                next_logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id:
                break
            tokens.append(next_token)

    return tokens


def tokens_to_events(tokens, id_to_token, pattern_id_to_name, patterns):
    """Convert tokens to musical events."""
    events = []
    current_beat = 0
    current_track = 0
    current_offset = 0

    for token_id in tokens:
        token = id_to_token.get(token_id, 'UNK')

        if token.startswith('BEAT_'):
            try:
                current_beat = int(token.split('_')[1])
            except:
                pass
        elif token.startswith('TRACK_'):
            try:
                current_track = int(token.split('_')[1])
            except:
                pass
        elif token.startswith('OFFSET_'):
            try:
                current_offset = int(token.split('_')[1])
            except:
                pass
        elif token.startswith('PATTERN_'):
            pattern_name = pattern_id_to_name.get(token)
            if pattern_name:
                notes = decode_pattern(
                    pattern_name, patterns,
                    transpose=current_offset,
                    time_offset=float(current_beat)
                )
                for note in notes:
                    note['gm_program'] = current_track
                    events.append(note)

    return events


def events_to_midi(events, output_path, tempo=120):
    """Convert events to MIDI file."""
    if not events:
        return False, {}

    # Normalize times to start from 0
    min_time = min(e['time'] for e in events)
    for e in events:
        e['time'] -= min_time

    # Group by program
    by_program = {}
    for e in events:
        gm = e['gm_program']
        if gm not in by_program:
            by_program[gm] = []
        by_program[gm].append(e)

    n_tracks = len(by_program)
    midi = MIDIFile(n_tracks, deinterleave=False)
    stats = {'tracks': {}, 'total_notes': 0}

    for track_idx, (gm_program, track_events) in enumerate(sorted(by_program.items())):
        track_name = GM_NAMES.get(gm_program, f'Program_{gm_program}')
        midi.addTrackName(track_idx, 0, track_name)
        midi.addTempo(track_idx, 0, tempo)

        channel = 9 if gm_program == 128 else track_idx % 16
        if gm_program != 128:
            midi.addProgramChange(track_idx, channel, 0, min(127, gm_program))

        for e in track_events:
            midi.addNote(track_idx, channel,
                        max(0, min(127, e['pitch'])),
                        max(0, e['time']),
                        max(0.05, e['duration']),
                        max(1, min(127, e['velocity'])))

        stats['tracks'][track_name] = len(track_events)
        stats['total_notes'] += len(track_events)

    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    return True, stats


# ============================================================================
# VALIDATION
# ============================================================================

def validate_and_analyze(events, tokens, id_to_token):
    """Validate and analyze the output."""
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    if not events:
        print("CRITICAL: No events generated!")
        return False

    pitches = [e['pitch'] for e in events]
    times = [e['time'] for e in events]
    velocities = [e['velocity'] for e in events]
    durations = [e['duration'] for e in events]

    # Token analysis
    token_types = {}
    for t in tokens:
        tok = id_to_token.get(t, 'UNK')
        prefix = tok.split('_')[0] if '_' in tok else tok
        token_types[prefix] = token_types.get(prefix, 0) + 1

    print("\nToken distribution:")
    for k, v in sorted(token_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")

    print(f"\nNote statistics:")
    print(f"  Total notes: {len(events)}")
    print(f"  Pitch range: {min(pitches)}-{max(pitches)} ({max(pitches)-min(pitches)} semitones)")
    print(f"  Time span: {min(times):.1f}-{max(times):.1f} beats ({max(times)-min(times):.1f} beats)")
    print(f"  Velocity range: {min(velocities)}-{max(velocities)} ({len(set(velocities))} unique)")
    print(f"  Duration range: {min(durations):.2f}-{max(durations):.2f}")

    # Per-instrument
    by_gm = {}
    for e in events:
        gm = e['gm_program']
        if gm not in by_gm:
            by_gm[gm] = []
        by_gm[gm].append(e)

    print(f"\nPer-instrument:")
    for gm, notes in sorted(by_gm.items()):
        name = GM_NAMES.get(gm, f'GM{gm}')
        gm_pitches = [n['pitch'] for n in notes]
        print(f"  {name}: {len(notes)} notes, pitch {min(gm_pitches)}-{max(gm_pitches)}")

    # Quality checks
    issues = []
    pattern_tokens = [id_to_token.get(t, '') for t in tokens if id_to_token.get(t, '').startswith('PATTERN_')]
    unique_patterns = set(pattern_tokens)

    if len(unique_patterns) < 5 and len(pattern_tokens) > 10:
        issues.append(f"Low variety: only {len(unique_patterns)} unique patterns in {len(pattern_tokens)}")

    if len(set(velocities)) < 3:
        issues.append("Flat dynamics (few unique velocities)")

    if max(pitches) - min(pitches) < 10:
        issues.append("Narrow pitch range")

    if len(by_gm) < 2:
        issues.append("Single instrument only")

    if issues:
        print("\nQuality issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No major quality issues detected")

    return len(issues) == 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/patternid/best_model.pt')
    parser.add_argument('--vocab', default='piece_level_vocab.json')
    parser.add_argument('--patterns', default='checkpoint_v55_pure_contour_1000files_patterns.json')
    parser.add_argument('--pattern-map', default='pattern_id_to_name.json')
    parser.add_argument('--output', default='generated_optimized.mid')
    parser.add_argument('--max-length', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--tempo', type=int, default=120)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    device = torch.device(args.device)

    print("="*60)
    print("OPTIMIZED MIDI GENERATION")
    print("="*60)
    sys.stdout.flush()

    # Load vocab
    print("\nLoading vocab...")
    sys.stdout.flush()
    with open(base_dir / args.vocab) as f:
        vocab = json.load(f)['vocab']
    id_to_token = {v: k for k, v in vocab.items()}
    print(f"  Vocab size: {len(vocab)}")

    # Load pattern ID mapping
    with open(base_dir / args.pattern_map) as f:
        pattern_id_to_name = json.load(f)
    print(f"  Pattern mappings: {len(pattern_id_to_name)}")
    sys.stdout.flush()

    # Load only needed patterns (optimized)
    patterns = load_needed_patterns(base_dir / args.patterns, pattern_id_to_name)

    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    checkpoint = torch.load(base_dir / args.checkpoint, map_location=device)

    if 'args' in checkpoint:
        model_args = checkpoint['args']
        vocab_size = checkpoint['vocab_size']
    else:
        model_args = {'d_model': 512, 'n_heads': 8, 'n_layers': 6, 'd_ff': 2048, 'max_length': 512}
        vocab_size = len(vocab)

    model = PatternIDTransformer(
        vocab_size=vocab_size,
        d_model=model_args.get('d_model', 512),
        n_heads=model_args.get('n_heads', 8),
        n_layers=model_args.get('n_layers', 6),
        d_ff=model_args.get('d_ff', 2048),
        dropout=0.0,
        max_len=model_args.get('max_length', 512)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    sys.stdout.flush()

    # Generate
    print("\n" + "="*60)
    print("GENERATING...")
    sys.stdout.flush()

    tokens = generate_tokens(model, vocab, device, args.max_length, args.temperature)
    print(f"  Generated {len(tokens)} tokens")

    token_strs = [id_to_token.get(t, '?') for t in tokens[:40]]
    print(f"  First 40: {' '.join(token_strs)}")
    sys.stdout.flush()

    # Convert to events
    events = tokens_to_events(tokens, id_to_token, pattern_id_to_name, patterns)
    print(f"  Decoded to {len(events)} notes")
    sys.stdout.flush()

    # Validate and analyze
    is_valid = validate_and_analyze(events, tokens, id_to_token)

    # Save MIDI
    output_path = base_dir / args.output
    success, stats = events_to_midi(events, output_path, args.tempo)

    if success:
        print(f"\n✓ Saved to: {output_path}")
        print(f"  Tracks: {stats['tracks']}")
        print(f"  Total notes: {stats['total_notes']}")
    else:
        print(f"\n✗ Failed to save MIDI")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
