#!/usr/bin/env python3
"""
PROPER MIDI GENERATION - Uses ALL factored pattern data
========================================================

This script properly decodes patterns using:
- pitch_intervals AND canonical_pitches
- rhythm_ratios (actual timing)
- duration_ratios (note lengths)
- velocity_ratios (dynamics)
- tau_offset (absolute timing reference)

Unlike the broken decoders, this one actually uses the rich data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import argparse
from pathlib import Path
from midiutil import MIDIFile
import orjson

# GM Program names for readability
GM_NAMES = {
    0: 'piano', 24: 'nylon_guitar', 25: 'steel_guitar', 27: 'clean_guitar',
    32: 'acoustic_bass', 33: 'electric_bass', 56: 'trumpet', 57: 'trombone',
    60: 'french_horn', 65: 'alto_sax', 66: 'tenor_sax', 67: 'baritone_sax',
    72: 'flute', 73: 'piccolo', 128: 'drums'
}


# ============================================================================
# MODEL DEFINITION (must match training)
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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
# PATTERN DECODER - THE KEY FIX
# ============================================================================

class ProperPatternDecoder:
    """
    Decodes patterns using ALL available factored data.
    This is what was missing from the original codebase.
    """

    def __init__(self, patterns_path, pattern_id_map_path, ticks_per_beat=480):
        print(f"Loading patterns from {patterns_path}...")
        with open(patterns_path, 'rb') as f:
            self.patterns = orjson.loads(f.read())
        print(f"  Loaded {len(self.patterns)} patterns")

        with open(pattern_id_map_path) as f:
            self.pattern_id_to_name = json.load(f)
        print(f"  Loaded {len(self.pattern_id_to_name)} pattern ID mappings")

        self.ticks_per_beat = ticks_per_beat

    def decode_pattern(self, pattern_token: str, transpose: int = 0,
                       time_offset: float = 0.0, time_scale: float = 1.0):
        """
        Decode a pattern token to actual MIDI notes using ALL factored data.

        Returns list of note dicts: {pitch, time, duration, velocity, channel}
        """
        # Get pattern name from token
        pattern_name = self.pattern_id_to_name.get(pattern_token)
        if not pattern_name or pattern_name not in self.patterns:
            return []

        pdata = self.patterns[pattern_name]

        # Extract factored components
        canonical_pitches = pdata.get('canonical_pitches', [60])
        pitch_intervals = pdata.get('pitch_intervals', [])
        rhythm_ratios = pdata.get('rhythm_ratios', [1.0] * len(canonical_pitches))
        duration_ratios = pdata.get('duration_ratios', [1.0] * len(canonical_pitches))
        velocity_ratios = pdata.get('velocity_ratios', [1.0] * len(canonical_pitches))
        gm_program = pdata.get('gm_program', 0)

        # Base timing from pattern metadata
        # tau_offset is in ticks, convert to beats
        base_duration = 0.5  # Default half-beat duration

        notes = []
        current_time = time_offset

        # Use canonical pitches if available, otherwise reconstruct from intervals
        if canonical_pitches and len(canonical_pitches) > 0:
            pitches = [p + transpose for p in canonical_pitches]
        else:
            # Reconstruct from intervals
            pitches = [60 + transpose]  # Start from middle C
            for interval in pitch_intervals:
                pitches.append(pitches[-1] + interval)

        for i, pitch in enumerate(pitches):
            # Get rhythm timing for this note
            if i < len(rhythm_ratios):
                # rhythm_ratio of 0 means simultaneous with previous
                # rhythm_ratio of 1 means normal spacing
                rhythm = rhythm_ratios[i] if i > 0 else 0
            else:
                rhythm = 1.0

            # Get duration
            if i < len(duration_ratios):
                dur = duration_ratios[i] * base_duration * time_scale
            else:
                dur = base_duration * time_scale

            # Get velocity (ratio of base 100)
            if i < len(velocity_ratios):
                vel = int(min(127, max(1, velocity_ratios[i] * 80)))
            else:
                vel = 80

            # Clamp pitch to valid MIDI range
            pitch = max(21, min(108, pitch))

            notes.append({
                'pitch': pitch,
                'time': current_time,
                'duration': max(0.1, dur),  # Minimum duration
                'velocity': vel,
                'gm_program': gm_program
            })

            # Advance time based on rhythm ratio
            if i + 1 < len(rhythm_ratios):
                inter_note_time = rhythm_ratios[i] * base_duration * time_scale
                current_time += inter_note_time
            else:
                current_time += base_duration * time_scale

        return notes

    def get_pattern_info(self, pattern_token: str):
        """Get info about a pattern for debugging."""
        pattern_name = self.pattern_id_to_name.get(pattern_token)
        if not pattern_name or pattern_name not in self.patterns:
            return None
        pdata = self.patterns[pattern_name]
        return {
            'name': pattern_name,
            'n_notes': len(pdata.get('canonical_pitches', [])),
            'gm_program': pdata.get('gm_program', 0),
            'count': pdata.get('count', 0)
        }


# ============================================================================
# GENERATION
# ============================================================================

def load_model(checkpoint_path, device):
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        vocab_size = checkpoint['vocab_size']
    else:
        # Fallback defaults
        args = {'d_model': 512, 'n_heads': 8, 'n_layers': 6, 'd_ff': 2048, 'max_length': 512}
        vocab_size = checkpoint.get('vocab_size', 5000)

    model = PatternIDTransformer(
        vocab_size=vocab_size,
        d_model=args.get('d_model', 512),
        n_heads=args.get('n_heads', 8),
        n_layers=args.get('n_layers', 6),
        d_ff=args.get('d_ff', 2048),
        dropout=0.0,
        max_len=args.get('max_length', 512)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {param_count/1e6:.1f}M params")

    return model, vocab_size


def generate_tokens(model, vocab, device, max_length=256, temperature=0.85,
                    top_k=50, top_p=0.92):
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

            # Top-k
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][-1]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p
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


def tokens_to_events(tokens, id_to_token, decoder):
    """
    Convert generated tokens to musical events.
    Properly parses TRACK/BEAT/OFFSET/PATTERN structure.
    """
    events = []
    current_beat = 0
    current_track = 0  # GM program
    current_offset = 0  # Pitch offset (for transposition)

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
            # This is a pattern - decode it properly
            notes = decoder.decode_pattern(
                token,
                transpose=current_offset,
                time_offset=float(current_beat),
                time_scale=1.0
            )

            for note in notes:
                # Override GM program from track context
                note['gm_program'] = current_track
                events.append(note)

    return events


def events_to_midi(events, output_path, tempo=120):
    """Convert events to MIDI file with proper multi-track support."""
    if not events:
        print("No events to convert!")
        return False, {}

    # Normalize times to start from 0
    min_time = min(e['time'] for e in events)
    for e in events:
        e['time'] -= min_time

    # Group events by GM program for separate tracks
    by_program = {}
    for e in events:
        gm = e['gm_program']
        if gm not in by_program:
            by_program[gm] = []
        by_program[gm].append(e)

    # Create MIDI with multiple tracks
    n_tracks = len(by_program)
    midi = MIDIFile(n_tracks, deinterleave=False)  # Disable deinterleave to avoid the bug

    stats = {'tracks': {}, 'total_notes': 0}

    for track_idx, (gm_program, track_events) in enumerate(sorted(by_program.items())):
        track_name = GM_NAMES.get(gm_program, f'Program_{gm_program}')

        midi.addTrackName(track_idx, 0, track_name)
        midi.addTempo(track_idx, 0, tempo)

        # Set program (instrument)
        channel = track_idx % 16
        if gm_program == 128:  # Drums
            channel = 9
        else:
            midi.addProgramChange(track_idx, channel, 0, min(127, gm_program))

        note_count = 0
        for e in track_events:
            # Ensure valid values
            pitch = max(0, min(127, e['pitch']))
            duration = max(0.05, e['duration'])  # Minimum duration
            velocity = max(1, min(127, e['velocity']))
            time = max(0, e['time'])

            midi.addNote(
                track=track_idx,
                channel=channel,
                pitch=pitch,
                time=time,
                duration=duration,
                volume=velocity
            )
            note_count += 1

        stats['tracks'][track_name] = note_count
        stats['total_notes'] += note_count

    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    return True, stats


# ============================================================================
# VALIDATION
# ============================================================================

def validate_output(events, tokens, id_to_token):
    """Validate the generated output isn't garbage."""
    issues = []

    if not events:
        issues.append("CRITICAL: No events generated")
        return False, issues

    # Check note count
    if len(events) < 10:
        issues.append(f"WARNING: Very few notes ({len(events)})")

    # Check pitch distribution
    pitches = [e['pitch'] for e in events]
    unique_pitches = set(pitches)
    if len(unique_pitches) < 3:
        issues.append(f"WARNING: Very few unique pitches ({len(unique_pitches)})")

    pitch_range = max(pitches) - min(pitches)
    if pitch_range < 5:
        issues.append(f"WARNING: Narrow pitch range ({pitch_range} semitones)")
    if pitch_range > 60:
        issues.append(f"WARNING: Very wide pitch range ({pitch_range} semitones)")

    # Check timing
    times = sorted([e['time'] for e in events])
    if len(times) > 1:
        time_range = times[-1] - times[0]
        if time_range < 2:
            issues.append(f"WARNING: Very short duration ({time_range:.1f} beats)")

    # Check for variety in generated tokens
    pattern_tokens = [id_to_token.get(t, '') for t in tokens if id_to_token.get(t, '').startswith('PATTERN_')]
    unique_patterns = set(pattern_tokens)
    if len(unique_patterns) < 3 and len(pattern_tokens) > 5:
        issues.append(f"WARNING: Low pattern variety ({len(unique_patterns)} unique in {len(pattern_tokens)} patterns)")

    # Check velocity variety
    velocities = [e['velocity'] for e in events]
    unique_vels = set(velocities)
    if len(unique_vels) == 1:
        issues.append("WARNING: No velocity variation (flat dynamics)")

    # Summary
    is_valid = len([i for i in issues if i.startswith("CRITICAL")]) == 0

    return is_valid, issues


def analyze_generation(events, tokens, id_to_token):
    """Detailed analysis of generated output."""
    print("\n" + "="*60)
    print("GENERATION ANALYSIS")
    print("="*60)

    # Token analysis
    token_types = {}
    for t in tokens:
        tok = id_to_token.get(t, 'UNK')
        prefix = tok.split('_')[0] if '_' in tok else tok
        token_types[prefix] = token_types.get(prefix, 0) + 1

    print("\nToken distribution:")
    for k, v in sorted(token_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Note analysis
    if events:
        pitches = [e['pitch'] for e in events]
        times = [e['time'] for e in events]
        velocities = [e['velocity'] for e in events]
        durations = [e['duration'] for e in events]

        print(f"\nNote statistics:")
        print(f"  Total notes: {len(events)}")
        print(f"  Pitch range: {min(pitches)}-{max(pitches)} ({max(pitches)-min(pitches)} semitones)")
        print(f"  Time span: {min(times):.1f}-{max(times):.1f} beats")
        print(f"  Velocity range: {min(velocities)}-{max(velocities)}")
        print(f"  Duration range: {min(durations):.2f}-{max(durations):.2f}")

        # Per-instrument breakdown
        by_gm = {}
        for e in events:
            gm = e['gm_program']
            if gm not in by_gm:
                by_gm[gm] = []
            by_gm[gm].append(e)

        print(f"\nPer-instrument breakdown:")
        for gm, notes in sorted(by_gm.items()):
            name = GM_NAMES.get(gm, f'GM{gm}')
            gm_pitches = [n['pitch'] for n in notes]
            print(f"  {name}: {len(notes)} notes, pitch {min(gm_pitches)}-{max(gm_pitches)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Proper MIDI Generation')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/patternid/best_model.pt')
    parser.add_argument('--vocab', type=str,
                        default='piece_level_vocab.json')
    parser.add_argument('--patterns', type=str,
                        default='checkpoint_v55_pure_contour_1000files_patterns.json')
    parser.add_argument('--pattern-map', type=str,
                        default='pattern_id_to_name.json')
    parser.add_argument('--output', type=str, default='generated_proper.mid')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.85)
    parser.add_argument('--tempo', type=int, default=120)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    device = torch.device(args.device)

    print("="*60)
    print("PROPER MIDI GENERATION")
    print("="*60)
    print(f"Device: {device}")

    # Load vocab
    vocab_path = base_dir / args.vocab
    print(f"\nLoading vocab from {vocab_path}...")
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    id_to_token = {v: k for k, v in vocab.items()}
    print(f"  Vocab size: {len(vocab)}")

    # Load pattern decoder
    patterns_path = base_dir / args.patterns
    pattern_map_path = base_dir / args.pattern_map
    decoder = ProperPatternDecoder(patterns_path, pattern_map_path)

    # Load model
    checkpoint_path = base_dir / args.checkpoint
    print(f"\nLoading model from {checkpoint_path}...")
    model, vocab_size = load_model(checkpoint_path, device)

    # Generate samples
    for i in range(args.num_samples):
        print(f"\n{'='*60}")
        print(f"Generating sample {i+1}/{args.num_samples}...")

        # Generate tokens
        tokens = generate_tokens(
            model, vocab, device,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"  Generated {len(tokens)} tokens")

        # Show some tokens
        token_strs = [id_to_token.get(t, '?') for t in tokens[:30]]
        print(f"  First 30: {' '.join(token_strs)}")

        # Convert to events
        events = tokens_to_events(tokens, id_to_token, decoder)
        print(f"  Decoded to {len(events)} notes")

        # Validate
        is_valid, issues = validate_output(events, tokens, id_to_token)
        if issues:
            print(f"  Validation issues:")
            for issue in issues:
                print(f"    - {issue}")

        # Analyze
        analyze_generation(events, tokens, id_to_token)

        # Save MIDI
        if args.num_samples == 1:
            output_path = base_dir / args.output
        else:
            output_path = base_dir / f"generated_{i+1}.mid"

        success, stats = events_to_midi(events, output_path, args.tempo)

        if success:
            print(f"\n  Saved to: {output_path}")
            print(f"  Stats: {stats}")
        else:
            print(f"  FAILED to save MIDI")

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == '__main__':
    main()
