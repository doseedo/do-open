"""
Generate MIDI from trained pattern ID model.
Decodes PATTERN_IDs back to intervals using pattern lookup.
"""
import argparse
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from midiutil import MIDIFile
import random

# GM Program to instrument name for better file naming
GM_NAMES = {
    0: 'piano', 24: 'nylon_guitar', 25: 'steel_guitar', 27: 'clean_guitar',
    32: 'acoustic_bass', 33: 'electric_bass', 56: 'trumpet', 57: 'trombone',
    60: 'french_horn', 65: 'alto_sax', 66: 'tenor_sax', 67: 'baritone_sax',
    72: 'flute', 73: 'piccolo', 128: 'drums'
}


def load_model(checkpoint_path, device):
    """Load trained model and vocab."""
    from train_patternid import PatternIDTransformer

    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    vocab_size = checkpoint['vocab_size']

    model = PatternIDTransformer(
        vocab_size=vocab_size,
        d_model=args['d_model'],
        n_heads=args['n_heads'],
        n_layers=args['n_layers'],
        d_ff=args['d_ff'],
        dropout=0.0,  # No dropout for generation
        max_len=args['max_length']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, args


def load_vocab_and_patterns(base_path):
    """Load vocabulary and pattern ID mappings."""
    # Load vocab
    with open(base_path / 'piece_level_vocab.json') as f:
        vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    id_to_token = {v: k for k, v in vocab.items()}

    # Load pattern ID to name mapping
    with open(base_path / 'pattern_id_to_name.json') as f:
        pattern_id_to_name = json.load(f)

    # Load original patterns for interval data
    import orjson
    with open(base_path / 'checkpoint_v55_pure_contour_1000files_patterns.json', 'rb') as f:
        patterns = orjson.loads(f.read())

    return vocab, id_to_token, pattern_id_to_name, patterns


def generate_sequence(model, vocab, device, max_length=256, temperature=0.8,
                      top_k=50, top_p=0.9, prompt_tokens=None):
    """Generate a sequence of tokens."""
    model.eval()

    if prompt_tokens is None:
        tokens = [vocab['BOS']]
    else:
        tokens = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

            if next_token == vocab['EOS']:
                break

    return tokens


def decode_to_events(tokens, id_to_token, pattern_id_to_name, patterns):
    """Convert generated tokens to musical events."""
    events = []
    current_beat = 0
    current_track = 0
    current_offset = 0

    for token_id in tokens:
        token = id_to_token.get(token_id, 'UNK')

        if token.startswith('BEAT_'):
            current_beat = int(token.split('_')[1])
        elif token.startswith('TRACK_'):
            current_track = int(token.split('_')[1])
        elif token.startswith('OFFSET_'):
            current_offset = int(token.split('_')[1])
        elif token.startswith('PATTERN_'):
            # Look up pattern
            pattern_name = pattern_id_to_name.get(token)
            if pattern_name and pattern_name in patterns:
                pattern_data = patterns[pattern_name]
                intervals = pattern_data.get('pitch_intervals', [])
                gm = pattern_data.get('gm_program', 0)

                events.append({
                    'beat': current_beat,
                    'track': current_track,
                    'gm': gm,
                    'offset': current_offset,
                    'intervals': intervals,
                    'pattern': pattern_name
                })

    return events


def events_to_midi(events, output_path, tempo=120):
    """Convert events to MIDI file."""
    if not events:
        print("No events to convert!")
        return False

    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0
    midi.addTempo(track, time, tempo)

    # Group events by instrument for better channel management
    gm_to_channel = {}
    next_channel = 0

    for event in events:
        gm = event['gm']
        if gm not in gm_to_channel:
            gm_to_channel[gm] = next_channel
            midi.addProgramChange(track, next_channel, 0, gm if gm != 128 else 0)
            next_channel = min(next_channel + 1, 15)

        ch = gm_to_channel[gm]
        intervals = event['intervals']
        beat = event['beat']
        offset = event['offset']

        # Convert intervals to notes
        base_pitch = 60 + offset  # Middle C + offset
        duration = 0.5  # Half beat per note

        # First note
        midi.addNote(track, ch, base_pitch, beat, duration, 100)

        # Subsequent notes based on intervals
        current_pitch = base_pitch
        current_time = beat + duration
        for interval in intervals:
            current_pitch += interval
            current_pitch = max(21, min(108, current_pitch))  # Keep in piano range
            midi.addNote(track, ch, current_pitch, current_time, duration, 100)
            current_time += duration

    # Write file
    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/patternid/best_model.pt')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='outputs/patternid_generation/')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--tempo', type=int, default=120)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_path = Path('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')

    # Load model
    print("Loading model...")
    model, model_args = load_model(base_path / args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    # Load vocab and patterns
    print("Loading vocabulary and patterns...")
    vocab, id_to_token, pattern_id_to_name, patterns = load_vocab_and_patterns(base_path)
    print(f"Vocab size: {len(vocab)}")
    print(f"Pattern IDs: {len(pattern_id_to_name)}")

    # Create output directory
    output_dir = base_path / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    for i in range(args.num_samples):
        print(f"\n=== Sample {i+1}/{args.num_samples} ===")

        # Generate tokens
        tokens = generate_sequence(
            model, vocab, device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

        print(f"Generated {len(tokens)} tokens")

        # Show some tokens
        token_names = [id_to_token.get(t, 'UNK') for t in tokens[:30]]
        print(f"First 30 tokens: {token_names}")

        # Decode to events
        events = decode_to_events(tokens, id_to_token, pattern_id_to_name, patterns)
        print(f"Decoded {len(events)} pattern events")

        # Count unique patterns and tracks
        unique_patterns = len(set(e['pattern'] for e in events))
        tracks_used = set(e['gm'] for e in events)
        print(f"Unique patterns: {unique_patterns}")
        print(f"Instruments: {[GM_NAMES.get(t, t) for t in tracks_used]}")

        # Convert to MIDI
        output_path = output_dir / f'sample_{i+1}.mid'
        if events_to_midi(events, output_path, args.tempo):
            print(f"Saved to {output_path}")
        else:
            print(f"Failed to create MIDI for sample {i+1}")

        # Save token sequence for analysis
        token_path = output_dir / f'sample_{i+1}_tokens.json'
        with open(token_path, 'w') as f:
            json.dump({
                'tokens': tokens,
                'token_names': [id_to_token.get(t, 'UNK') for t in tokens],
                'n_events': len(events),
                'n_unique_patterns': unique_patterns,
                'instruments': list(tracks_used)
            }, f, indent=2)

    print(f"\nGeneration complete. Output saved to {output_dir}")


if __name__ == '__main__':
    main()
