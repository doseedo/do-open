#!/usr/bin/env python3
"""
FIXED MIDI GENERATION v2
========================
Correct interpretation of token semantics:
- BEAT_X: Absolute beat position (0-63, wraps)
- TRACK_X: Sets current GM program for subsequent patterns
- OFFSET_X: Pitch transposition for next pattern
- PATTERN_X: Place pattern at current beat with current track/offset
"""
import torch
import torch.nn as nn
import json
import sys
import os
import math
from midiutil import MIDIFile

os.chdir('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')

print("=" * 60)
print("FIXED MIDI GENERATION v2")
print("=" * 60)

# Load vocab
with open('piece_level_vocab.json') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']

id_to_token = {v: k for k, v in vocab.items()}

with open('pattern_id_to_name.json') as f:
    id_to_name = json.load(f)

# Load patterns
with open('needed_patterns.json') as f:
    patterns = json.load(f)

print(f"Vocab: {len(vocab)}, Patterns: {len(patterns)}")

# Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class PatternIDTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        return self.fc_out(self.transformer(x, mask=mask, is_causal=True))

# Load model
model = PatternIDTransformer(len(vocab), max_len=512)
ckpt = torch.load('checkpoints/patternid/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Model loaded")

# Generate
def generate(model, vocab, max_length=400, temperature=0.95, min_length=100):
    id_to_token = {v: k for k, v in vocab.items()}
    eos_id = vocab.get('EOS', -1)
    pad_id = vocab.get('PAD', 0)

    current = torch.tensor([[vocab['BOS']]], dtype=torch.long)
    generated = [vocab['BOS']]

    with torch.no_grad():
        for step in range(max_length):
            logits = model(current)[0, -1, :] / temperature

            # Suppress EOS until min_length
            if step < min_length:
                logits[eos_id] = -float('inf')
                logits[pad_id] = -float('inf')

            probs = torch.softmax(logits, dim=-1)

            # Top-p sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum > 0.92
            mask[0] = False
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()

            next_token = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
            generated.append(next_token)
            current = torch.cat([current, torch.tensor([[next_token]])], dim=1)

            token_str = id_to_token.get(next_token, '')
            if token_str in ['EOS', 'PAD'] and step >= min_length:
                break
            if current.size(1) > 500:
                current = current[:, -500:]

    return [id_to_token.get(t, f'UNK_{t}') for t in generated]

print("\nGenerating...")
tokens = generate(model, vocab, max_length=500, temperature=0.95, min_length=200)
print(f"Generated {len(tokens)} tokens")
print(f"First 30: {tokens[:30]}")

# CORRECT DECODE
print("\nDecoding with correct semantics...")

events = []
current_beat = 0
current_track = 0  # GM program
current_offset = 0  # Pitch offset
beat_wrap_count = 0  # Track how many times we've wrapped past beat 63
track_times = {}  # Track current time position per instrument

for i, token in enumerate(tokens):
    if token.startswith('BEAT_'):
        new_beat = int(token.split('_')[1])
        # Detect wrap-around
        if new_beat < current_beat - 32:  # Wrapped
            beat_wrap_count += 1
        current_beat = new_beat

    elif token.startswith('TRACK_'):
        current_track = int(token.split('_')[1])

    elif token.startswith('OFFSET_'):
        current_offset = int(token.split('_')[1]) - 6  # Center around 0

    elif token.startswith('PATTERN_'):
        pattern_id = token
        if pattern_id in id_to_name:
            pattern_name = id_to_name[pattern_id]
            if pattern_name in patterns:
                p = patterns[pattern_name]
                pitches = p.get('canonical_pitches', [])
                rhythm = p.get('rhythm_ratios', [])
                durations = p.get('duration_ratios', [])
                velocities = p.get('velocity_ratios', [])

                # Calculate absolute time
                abs_beat = (beat_wrap_count * 64) + current_beat
                time_offset = abs_beat * 0.5  # Each beat unit = 0.5 beats

                # KEY FIX: Track time per instrument to prevent stacking
                if current_track not in track_times:
                    track_times[current_track] = time_offset
                else:
                    # If we'd go backwards in time for this track, advance instead
                    if time_offset < track_times[current_track]:
                        time_offset = track_times[current_track]

                for j, pitch in enumerate(pitches):
                    actual_pitch = max(0, min(127, pitch + current_offset))
                    dur = durations[j] * 0.5 if j < len(durations) else 0.3
                    dur = max(0.1, min(2.0, dur))
                    vel = velocities[j] * 80 if j < len(velocities) else 80
                    vel = int(max(40, min(120, vel)))

                    events.append({
                        'pitch': actual_pitch,
                        'time': time_offset,
                        'duration': dur,
                        'velocity': vel,
                        'gm_program': current_track
                    })

                    # Advance time within pattern
                    if j < len(rhythm):
                        time_offset += max(0.1, rhythm[j] * 0.5)
                    else:
                        time_offset += 0.25

                # Update track's current time
                track_times[current_track] = time_offset

print(f"Decoded {len(events)} notes")

if not events:
    print("ERROR: No events!")
    sys.exit(1)

# Sort by time
events.sort(key=lambda x: x['time'])

# Normalize time
min_time = events[0]['time']
for e in events:
    e['time'] -= min_time

# Group by program
by_prog = {}
for e in events:
    prog = e['gm_program']
    if prog not in by_prog:
        by_prog[prog] = []
    by_prog[prog].append(e)

print(f"Tracks: {sorted(by_prog.keys())}")

# Create MIDI
midi = MIDIFile(len(by_prog), deinterleave=False)  # Fix MIDI bug
tempo = 120

gm_names = {0: 'Piano', 32: 'Acoustic Bass', 33: 'Electric Bass', 56: 'Trumpet',
            57: 'Trombone', 60: 'French Horn', 65: 'Alto Sax', 66: 'Tenor Sax',
            67: 'Baritone Sax', 73: 'Flute', 128: 'Drums'}

for track_idx, (prog, track_events) in enumerate(sorted(by_prog.items())):
    channel = 9 if prog >= 128 else track_idx % 9
    if channel >= 9 and prog < 128:
        channel = (channel + 1) % 9

    midi.addTempo(track_idx, 0, tempo)
    midi.addProgramChange(track_idx, channel, 0, min(127, prog))

    for e in track_events:
        midi.addNote(track_idx, channel, e['pitch'], e['time'], e['duration'], e['velocity'])

    name = gm_names.get(prog, f'GM{prog}')
    print(f"  {name}: {len(track_events)} notes, time {track_events[0]['time']:.1f}-{track_events[-1]['time']:.1f}")

# Save
with open('generated_v2.mid', 'wb') as f:
    midi.writeFile(f)
print(f"\nSaved: generated_v2.mid")

# REAL EVALUATION
print("\n" + "=" * 60)
print("REAL EVALUATION")
print("=" * 60)

# 1. Temporal density
total_duration = events[-1]['time'] - events[0]['time']
notes_per_beat = len(events) / max(1, total_duration)
print(f"Duration: {total_duration:.1f} beats")
print(f"Density: {notes_per_beat:.2f} notes/beat")

# 2. Polyphony - notes at same time
from collections import Counter
time_buckets = Counter(round(e['time'], 1) for e in events)
max_poly = max(time_buckets.values())
avg_poly = sum(time_buckets.values()) / len(time_buckets)
print(f"Max polyphony: {max_poly}")
print(f"Avg polyphony: {avg_poly:.1f}")

# 3. Instrument balance
print(f"Instrument balance:")
for prog, evts in sorted(by_prog.items()):
    pct = len(evts) / len(events) * 100
    print(f"  {gm_names.get(prog, f'GM{prog}')}: {pct:.0f}%")

# 4. Pitch coherence per track
print(f"Pitch coherence:")
for prog, evts in sorted(by_prog.items()):
    pitches = [e['pitch'] for e in evts]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
    if intervals:
        avg_interval = sum(abs(i) for i in intervals) / len(intervals)
        big_jumps = sum(1 for i in intervals if abs(i) > 12)
        print(f"  {gm_names.get(prog, f'GM{prog}')}: avg interval {avg_interval:.1f}, big jumps {big_jumps}")

# 5. Structure - look for repetition
print(f"\nStructure check:")
pitch_seq = tuple(e['pitch'] for e in events[:200])
patterns_4 = Counter(pitch_seq[i:i+4] for i in range(len(pitch_seq)-3))
repeated = sum(1 for c in patterns_4.values() if c > 1)
print(f"  4-note patterns that repeat: {repeated}")

# 6. Gap analysis
times = sorted(set(round(e['time'], 1) for e in events))
gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
if gaps:
    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)
    big_gaps = sum(1 for g in gaps if g > 4)
    print(f"  Avg time gap: {avg_gap:.2f} beats")
    print(f"  Max gap: {max_gap:.1f} beats")
    print(f"  Gaps > 4 beats: {big_gaps}")

print("\n" + "=" * 60)
