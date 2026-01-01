#!/usr/bin/env python3
"""
Generate and evaluate compositional sequences from trained model.

Tests:
1. Generate with different instrument prompts
2. Decode INTERVAL+COMPOSE to MIDI
3. Analyze interval distributions
4. Check for novelty vs memorization
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from compositional_model import create_model

# MIDI writing
try:
    import midiutil
    HAS_MIDIUTIL = True
except ImportError:
    HAS_MIDIUTIL = False
    print("Warning: midiutil not installed, will skip MIDI export")


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = create_model(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config.get('max_length', 1024) + 10,
        pad_id=0,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def load_vocab(data_path):
    """Load vocabulary from training data."""
    data = torch.load(data_path)
    return data['vocab'], data['id_to_token']


def generate_sequence(model, prompt, vocab, max_new_tokens=200,
                      temperature=0.8, top_k=40, top_p=0.9, device='cuda'):
    """Generate a sequence from prompt."""
    model.eval()

    eos_id = vocab.get('EOS', 2)
    generated = torch.tensor([prompt], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_id:
                break

    return generated[0].tolist()


def decode_to_intervals(token_ids, id_to_token):
    """Extract interval sequences from generated tokens."""
    intervals = []
    current_pattern = []

    for tid in token_ids:
        token = id_to_token.get(tid, f'UNK_{tid}')

        if token.startswith('INTERVAL_'):
            # Parse interval value
            val_str = token.replace('INTERVAL_', '')
            try:
                val = int(val_str)
                current_pattern.append(val)
            except:
                pass
        elif token == 'COMPOSE':
            if current_pattern:
                intervals.append(current_pattern)
                current_pattern = []
        elif token in ['EOS', 'BOS']:
            if current_pattern:
                intervals.append(current_pattern)
                current_pattern = []

    if current_pattern:
        intervals.append(current_pattern)

    return intervals


def intervals_to_pitches(interval_patterns, start_pitch=60):
    """Convert interval patterns to absolute pitches."""
    all_pitches = []
    current_pitch = start_pitch

    for pattern in interval_patterns:
        pattern_pitches = [current_pitch]
        for interval in pattern:
            current_pitch += interval
            current_pitch = max(21, min(108, current_pitch))  # Piano range
            pattern_pitches.append(current_pitch)
        all_pitches.append(pattern_pitches)

    return all_pitches


def create_midi(pitch_patterns, output_path, tempo=120, ticks_per_beat=480):
    """Create MIDI file from pitch patterns."""
    if not HAS_MIDIUTIL:
        return False

    from midiutil import MIDIFile

    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)

    time = 0  # In beats
    duration = 0.5  # Half beat per note

    for pattern in pitch_patterns:
        for pitch in pattern:
            midi.addNote(0, 0, pitch, time, duration, 100)
            time += duration
        time += 0.25  # Small gap between patterns

    with open(output_path, 'wb') as f:
        midi.writeFile(f)

    return True


def analyze_intervals(all_intervals):
    """Analyze interval distribution."""
    flat_intervals = []
    pattern_lengths = []

    for seq_intervals in all_intervals:
        for pattern in seq_intervals:
            flat_intervals.extend(pattern)
            pattern_lengths.append(len(pattern))

    interval_counts = Counter(flat_intervals)

    return {
        'interval_histogram': dict(interval_counts.most_common(20)),
        'total_intervals': len(flat_intervals),
        'unique_intervals': len(interval_counts),
        'avg_pattern_length': sum(pattern_lengths) / len(pattern_lengths) if pattern_lengths else 0,
        'pattern_count': len(pattern_lengths),
    }


def check_novelty(generated_intervals, training_data_path):
    """Check if generated patterns are novel or memorized."""
    # Load training sequences
    data = torch.load(training_data_path)
    vocab = data['vocab']
    id_to_token = data['id_to_token']

    # Extract interval patterns from training
    training_patterns = set()
    for seq in data['sequences'][:100]:  # Sample
        intervals = decode_to_intervals(seq, id_to_token)
        for pattern in intervals:
            training_patterns.add(tuple(pattern))

    # Check generated
    novel_count = 0
    memorized_count = 0

    for seq_intervals in generated_intervals:
        for pattern in seq_intervals:
            if tuple(pattern) in training_patterns:
                memorized_count += 1
            else:
                novel_count += 1

    return {
        'novel_patterns': novel_count,
        'memorized_patterns': memorized_count,
        'novelty_ratio': novel_count / (novel_count + memorized_count) if (novel_count + memorized_count) > 0 else 0,
        'training_patterns_sampled': len(training_patterns),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate compositional sequences')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/checkpoints/compositional_piece/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/piece_compositional_data.pt',
                        help='Path to training data (for vocab)')
    parser.add_argument('--output-dir', type=str,
                        default='/home/arlo/do-repo/midi_generator/1_approaches/transform_based/outputs/compositional_generation',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples per prompt')
    parser.add_argument('--max-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and vocab
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, device)
    print(f"Model loaded: {config['vocab_size']} vocab, {config['d_model']}d, {config['n_layers']} layers")

    print("\nLoading vocabulary...")
    vocab, id_to_token = load_vocab(args.data_path)

    # Define prompts for different instruments
    prompts = {
        'alto_sax': [vocab['BOS'], vocab['BEAT_0'], vocab['TRACK_65'], vocab['OFFSET_0']],
        'bass': [vocab['BOS'], vocab['BEAT_0'], vocab['TRACK_33'], vocab['OFFSET_0']],
        'flute': [vocab['BOS'], vocab['BEAT_0'], vocab['TRACK_73'], vocab['OFFSET_0']],
        'french_horn': [vocab['BOS'], vocab['BEAT_0'], vocab['TRACK_60'], vocab['OFFSET_0']],
        'baritone_sax': [vocab['BOS'], vocab['BEAT_0'], vocab['TRACK_67'], vocab['OFFSET_0']],
        'piano': [vocab['BOS'], vocab['BEAT_0'], vocab['TRACK_0'], vocab['OFFSET_0']],
    }

    # Generate samples
    print("\n" + "="*60)
    print("Generating samples...")
    print("="*60)

    all_generated = {}
    all_intervals = []

    for name, prompt in prompts.items():
        print(f"\n--- {name.upper()} ---")
        prompt_tokens = [id_to_token[t] for t in prompt]
        print(f"Prompt: {' '.join(prompt_tokens)}")

        samples = []
        for i in range(args.num_samples):
            tokens = generate_sequence(
                model, prompt, vocab,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device
            )
            samples.append(tokens)

            # Decode and show first few
            if i < 2:
                decoded = [id_to_token.get(t, '?') for t in tokens[:30]]
                print(f"  Sample {i+1}: {' '.join(decoded)}...")

        all_generated[name] = samples

        # Extract intervals
        for seq in samples:
            intervals = decode_to_intervals(seq, id_to_token)
            all_intervals.append(intervals)

        # Create MIDI for first sample
        if samples:
            intervals = decode_to_intervals(samples[0], id_to_token)
            pitches = intervals_to_pitches(intervals)
            midi_path = output_dir / f'{name}_sample.mid'
            if create_midi(pitches, midi_path):
                print(f"  Saved MIDI: {midi_path}")

    # Analyze intervals
    print("\n" + "="*60)
    print("Interval Analysis")
    print("="*60)

    stats = analyze_intervals(all_intervals)
    print(f"\nTotal intervals generated: {stats['total_intervals']}")
    print(f"Unique interval values: {stats['unique_intervals']}")
    print(f"Average pattern length: {stats['avg_pattern_length']:.1f}")
    print(f"Total patterns: {stats['pattern_count']}")

    print("\nTop 15 intervals:")
    for interval, count in sorted(stats['interval_histogram'].items(), key=lambda x: -x[1])[:15]:
        pct = count / stats['total_intervals'] * 100
        bar = '#' * int(pct)
        print(f"  {interval:+3d}: {count:5d} ({pct:5.1f}%) {bar}")

    # Check novelty
    print("\n" + "="*60)
    print("Novelty Analysis")
    print("="*60)

    novelty = check_novelty(all_intervals, args.data_path)
    print(f"\nNovel patterns: {novelty['novel_patterns']}")
    print(f"Memorized patterns: {novelty['memorized_patterns']}")
    print(f"Novelty ratio: {novelty['novelty_ratio']:.1%}")
    print(f"(Compared against {novelty['training_patterns_sampled']} training patterns)")

    # Save results
    results = {
        'config': config,
        'stats': stats,
        'novelty': novelty,
        'prompts': {k: [id_to_token[t] for t in v] for k, v in prompts.items()},
    }

    with open(output_dir / 'analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # Show sample decoded sequences
    print("\n" + "="*60)
    print("Sample Decoded Sequences (intervals)")
    print("="*60)

    for name in list(prompts.keys())[:3]:
        print(f"\n{name}:")
        if all_generated[name]:
            intervals = decode_to_intervals(all_generated[name][0], id_to_token)
            for i, pattern in enumerate(intervals[:5]):
                print(f"  Pattern {i+1}: {pattern}")


if __name__ == '__main__':
    main()
